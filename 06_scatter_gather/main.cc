#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* source, float* dest, const int* indices, int size) {
    for (int i = 0; i < size; ++i) {
        dest[i] = source[indices[i]]; // Gather
    }
}

struct GatherTest : public ProblemTest<1> {
    std::vector<float> h_source;
    std::vector<float> h_dest;
    std::vector<float> h_dest_ref;
    std::vector<int> h_indices;

    float *d_source = nullptr;
    float *d_dest = nullptr;
    int *d_indices = nullptr;

    GatherTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int n = size.dims[0];
        h_source.resize(n);
        h_dest.assign(n, 0.0f);
        h_dest_ref.assign(n, 0.0f);
        h_indices.resize(n);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        for (int i = 0; i < n; ++i) {
            h_source[i] = dist(gen);
            h_indices[i] = i;
        }
        std::shuffle(h_indices.begin(), h_indices.end(), gen);

        if (check) cpu_baseline(h_source.data(), h_dest_ref.data(), h_indices.data(), n);
    }

    void setup_reference() override {
        int n = size.dims[0];
        cudaMalloc(&d_source, n * sizeof(float));
        cudaMalloc(&d_dest, n * sizeof(float));
        cudaMalloc(&d_indices, n * sizeof(int));
        
        cudaMemcpy(d_source, h_source.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_dest, 0, n * sizeof(float));
    }

    LaunchMetrics launch_reference() override {
        return launch_reference_gather(d_source, d_dest, d_indices, size.dims[0]);
    }

    void setup_student() override {
        int n = size.dims[0];
        cudaMalloc(&d_source, n * sizeof(float));
        cudaMalloc(&d_dest, n * sizeof(float));
        cudaMalloc(&d_indices, n * sizeof(int));
        
        cudaMemcpy(d_source, h_source.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_dest, 0, n * sizeof(float));
    }

    LaunchMetrics launch_student() override {
        return launch_gather(d_source, d_dest, d_indices, size.dims[0]);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_dest.data(), d_dest, size.dims[0] * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_dest_ref.data(), h_dest.data(), size.dims[0], 1e-5f);
    }

    void print_mismatch() override {
        std::cout << "\n--- Expected Output ---\n";
        for (int i = 0; i < std::min(size.dims[0], 10LL); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_dest_ref[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n--- Actual Output ---\n";
        for (int i = 0; i < std::min(size.dims[0], 10LL); ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << h_dest[i];
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }

    double get_bandwidth_bytes() override {
        return 3.0 * size.dims[0] * sizeof(float); // float read, int read, float write
    }

    void teardown() override {
        if (d_source) cudaFree(d_source);
        if (d_dest) cudaFree(d_dest);
        if (d_indices) cudaFree(d_indices);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {1048576}, {16777216}, {67108864}
    };

    run_test_suite<1, GatherTest>("Exercise 06: Scatter/Gather", config, correctness_sizes, perf_sizes);
    return 0;
}
