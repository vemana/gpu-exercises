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

void cpu_baseline(const float* a, const float* b, float* c, long long n) {
    for (long long row = 0; row < n; ++row) {
        for (long long col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (long long p = 0; p < n; ++p) {
                sum += a[row * n + p] * b[p * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

struct GemmTest : public ProblemTest<1> {
    std::vector<float> h_a;
    std::vector<float> h_b;
    std::vector<float> h_c;
    std::vector<float> h_c_ref;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    GemmTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        long long n = size.dims[0];
        h_a.resize(n * n);
        h_b.resize(n * n);
        h_c.assign(n * n, 0.0f);
        h_c_ref.assign(n * n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (long long i = 0; i < n * n; ++i) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }
        if (check) cpu_baseline(h_a.data(), h_b.data(), h_c_ref.data(), n);
    }

    void setup_reference() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * n * sizeof(float));
        cudaMalloc(&d_b, n * n * sizeof(float));
        cudaMalloc(&d_c, n * n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_gemm(d_a, d_b, d_c, size.dims[0]);
    }

    void setup_student() override {
        long long n = size.dims[0];
        cudaMalloc(&d_a, n * n * sizeof(float));
        cudaMalloc(&d_b, n * n * sizeof(float));
        cudaMalloc(&d_c, n * n * sizeof(float));
        
        cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, n * n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_gemm(d_a, d_b, d_c, size.dims[0]);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0];
        cudaMemcpy(h_c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_c_ref.data(), h_c.data(), n * n, 1e-2f);
    }

    void print_mismatch() override {
        long long n = size.dims[0];
        print_array("Input (a)", h_a.data(), n * n);
        print_array("Input (b)", h_b.data(), n * n);
        print_array("Expected Output", h_c_ref.data(), n * n);
        print_array("Actual Output", h_c.data(), n * n);
    }

    double get_bandwidth_bytes() override {
        long long n = size.dims[0];
        return 3.0 * n * n * sizeof(float); // Theoretical minimum: Read A, Read B, Write C
    }

    void teardown() override {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // CPU baseline is O(N^3), so keep correctness sizes small
    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {255}, {256}, {257}, {512}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {1024}, {2048}, {4096}
    };

    run_test_suite<1, GemmTest>("Exercise 07: Matrix Multiplication (GEMM)", config, correctness_sizes, perf_sizes);
    return 0;
}
