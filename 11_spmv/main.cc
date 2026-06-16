#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, int num_rows) {
    for (int i = 0; i < num_rows; ++i) {
        float sum = 0.0f;
        int start = row_offsets[i];
        int end = row_offsets[i + 1];
        for (int j = start; j < end; ++j) {
            sum += values[j] * x[col_indices[j]];
        }
        y[i] = sum;
    }
}

struct SpMVTest : public ProblemTest<1> {
    std::vector<float> h_values;
    std::vector<int> h_col_indices;
    std::vector<int> h_row_offsets;
    std::vector<float> h_x;
    std::vector<float> h_y;
    std::vector<float> h_y_ref;
    int nnz = 0;

    float *d_values = nullptr;
    int *d_col_indices = nullptr;
    int *d_row_offsets = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    SpMVTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_row_offsets.resize(n + 1, 0);
        h_x.resize(n);
        h_y.assign(n, 0.0f);
        h_y_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
        
        // Generate a random sparse matrix with avg 4 non-zeros per row
        h_row_offsets[0] = 0;
        for (int i = 0; i < n; ++i) {
            h_x[i] = val_dist(gen);
            
            // random number of non-zeros, mean 4
            int row_nnz = 2 + (gen() % 5); 
            if (row_nnz > n) row_nnz = n;
            
            for (int j = 0; j < row_nnz; ++j) {
                h_values.push_back(val_dist(gen));
                h_col_indices.push_back(gen() % n);
            }
            h_row_offsets[i + 1] = h_values.size();
        }
        nnz = h_values.size();

        if (check) cpu_baseline(h_values.data(), h_col_indices.data(), h_row_offsets.data(), h_x.data(), h_y_ref.data(), n);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMalloc(&d_col_indices, nnz * sizeof(int));
        cudaMalloc(&d_row_offsets, (n + 1) * sizeof(int));
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        
        cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_offsets, h_row_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, n * sizeof(float));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_spmv(d_values, d_col_indices, d_row_offsets, d_x, d_y, size.dims[0]);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMalloc(&d_col_indices, nnz * sizeof(int));
        cudaMalloc(&d_row_offsets, (n + 1) * sizeof(int));
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        
        cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_offsets, h_row_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, n * sizeof(float));
    }

    LaunchMetrics launch_student() override {
        return launch_spmv(d_values, d_col_indices, d_row_offsets, d_x, d_y, size.dims[0]);
    }

    CorrectnessResult verify() override {
        int n = size.dims[0];
        cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_y_ref.data(), h_y.data(), n, 1e-4f);
    }

    void print_mismatch() override {
        int n = size.dims[0];
        std::cout << "\n--- Expected Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << h_y_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output (First 10) ---\n";
        for (int i = 0; i < std::min(n, 10); ++i) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << h_y[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        int n = size.dims[0];
        // Read values, col_indices, row_offsets, X vector (approx). Write Y vector.
        return nnz * sizeof(float) + nnz * sizeof(int) + (n + 1) * sizeof(int) + n * sizeof(float) + n * sizeof(float);
    }

    void teardown() override {
        if (d_values) cudaFree(d_values);
        if (d_col_indices) cudaFree(d_col_indices);
        if (d_row_offsets) cudaFree(d_row_offsets);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {1048576}, {4194304}, {16777216}
    };

    run_test_suite<1, SpMVTest>("Exercise 11: Sparse Matrix-Vector Multiplication (SpMV)", config, correctness_sizes, perf_sizes);
    return 0;
}
