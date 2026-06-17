#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

void generate_laplacian_2d(int nx, int ny, std::vector<int>& row_ptr, std::vector<int>& col_ind, std::vector<float>& values) {
    int N = nx * ny;
    row_ptr.push_back(0);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int row = y * nx + x;
            
            // 5-point stencil (Laplacian)
            if (y > 0) { col_ind.push_back(row - nx); values.push_back(-1.0f); }
            if (x > 0) { col_ind.push_back(row - 1); values.push_back(-1.0f); }
            
            col_ind.push_back(row); values.push_back(4.0f);
            
            if (x < nx - 1) { col_ind.push_back(row + 1); values.push_back(-1.0f); }
            if (y < ny - 1) { col_ind.push_back(row + nx); values.push_back(-1.0f); }
            
            row_ptr.push_back(col_ind.size());
        }
    }
}

void cpu_spmv(const std::vector<int>& row_ptr, const std::vector<int>& col_ind, const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y, int N) {
    for (int row = 0; row < N; ++row) {
        double sum = 0.0;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int j = start; j < end; ++j) {
            sum += (double)values[j] * (double)x[col_ind[j]];
        }
        y[row] = (float)sum;
    }
}

float cpu_dot(const std::vector<float>& a, const std::vector<float>& b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

void cpu_cg(const std::vector<int>& row_ptr, const std::vector<int>& col_ind, const std::vector<float>& values, const std::vector<float>& b, std::vector<float>& x, int N, int num_iters) {
    std::vector<float> r(N, 0.0f);
    std::vector<float> p(N, 0.0f);
    std::vector<float> Ap(N, 0.0f);
    
    // r = b - Ax
    cpu_spmv(row_ptr, col_ind, values, x, Ap, N);
    for (int i = 0; i < N; ++i) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }
    
    float r_dot_r = cpu_dot(r, r, N);
    
    for (int iter = 0; iter < num_iters; ++iter) {
        cpu_spmv(row_ptr, col_ind, values, p, Ap, N);
        float p_Ap = cpu_dot(p, Ap, N);
        
        float alpha = r_dot_r / p_Ap;
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        float r_dot_r_new = cpu_dot(r, r, N);
        float beta = r_dot_r_new / r_dot_r;
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        
        r_dot_r = r_dot_r_new;
    }
}

const int NUM_ITERS = 50;

struct CGTest : public ProblemTest<1> {
    int N;
    int nnz;
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_ind;
    std::vector<float> h_values;
    std::vector<float> h_b;
    std::vector<float> h_x_initial;
    std::vector<float> h_x;
    std::vector<float> h_x_ref;

    int *d_row_ptr = nullptr;
    int *d_col_ind = nullptr;
    float *d_values = nullptr;
    float *d_b = nullptr;
    float *d_x = nullptr;

    CGTest(const TestSize<1>& size) : ProblemTest<1>(size) {
        // We use size.dims[0] as the grid size for the 2D Laplacian (nx = ny = sqrt(size))
        int nx = (int)std::sqrt(size.dims[0]);
        N = nx * nx; // ensure perfect square
    }

    void generate_test_data(bool check) override {
        generate_laplacian_2d(std::sqrt(N), std::sqrt(N), h_row_ptr, h_col_ind, h_values);
        nnz = h_col_ind.size();
        
        h_b.resize(N, 1.0f);
        h_x_initial.resize(N, 0.0f);
        h_x.resize(N, 0.0f);
        h_x_ref.resize(N, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < N; ++i) {
            h_b[i] = dist(gen);
        }

        if (check) {
            h_x_ref = h_x_initial;
            cpu_cg(h_row_ptr, h_col_ind, h_values, h_b, h_x_ref, N, NUM_ITERS);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
        cudaMalloc(&d_col_ind, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_x, N * sizeof(float));
        
        cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_cg_reference(d_row_ptr, d_col_ind, d_values, d_b, d_x, N, nnz, NUM_ITERS);
    }

    void teardown_reference() override {
        cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);
        cudaFree(d_b); cudaFree(d_x);
    }

    void setup_student() override {
        cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
        cudaMalloc(&d_col_ind, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_x, N * sizeof(float));
        
        cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x_initial.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_cg(d_row_ptr, d_col_ind, d_values, d_b, d_x, N, nnz, NUM_ITERS);
    }

    void teardown_student() override {
        cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);
        cudaFree(d_b); cudaFree(d_x);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
        // Iterative solvers accumulate errors; tolerance should scale with iterations.
        for (int i = 0; i < N; ++i) {
            float diff = std::abs(h_x_ref[i] - h_x[i]);
            float magnitude = std::abs(h_x_ref[i]) + 1e-5f;
            if (diff > 1e-1f && (diff / magnitude) > 1e-2f) {
                return {false, i, h_x_ref[i], h_x[i]};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min(N, 20); ++i) {
            std::cout << "Idx " << i << ": Expected " << h_x_ref[i] << " | Actual " << h_x[i] << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        // CG memory bandwidth per iteration roughly:
        // SpMV: nnz * (4 bytes idx + 4 bytes val) + nnz * 4 bytes x_read + N * 4 bytes Ap_write = 12*nnz + 4*N
        // dot p_Ap: 2 * N * 4 = 8*N
        // update_x_r: N * 4 * (2 read, 2 write) = 16*N
        // dot r_new: N * 4 = 4*N
        // update_p: N * 4 * (2 read, 1 write) = 12*N
        // Total per iter: ~ 12*nnz + 44*N bytes
        return (double)NUM_ITERS * (12.0 * nnz + 44.0 * N);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Grid sizes: nx*nx
    std::vector<TestSize<1>> correctness_sizes = {
        {{16 * 16}}, {{32 * 32}}, {{64 * 64}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{128 * 128}}, {{256 * 256}}, {{512 * 512}}
    };

    run_test_suite<1, CGTest>("Exercise 28: Conjugate Gradient", config, correctness_sizes, perf_sizes);
    return 0;
}
