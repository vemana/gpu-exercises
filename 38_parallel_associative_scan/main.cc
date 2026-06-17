#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "../utils/utils.h"
#include "kernel.h"
#include "reference_kernel.h"

// CPU Baseline: Computes the recurrence h[t] = A[t]*h[t-1] + B[t]*x[t] sequentially.
// This strict loop is what limits RNN speeds on CPUs.
void cpu_scan(const float* A, const float* B, const float* x, float* h, int N) {
    float prev_h = 0.0f;
    for (int t = 0; t < N; ++t) {
        float h_t = A[t] * prev_h + B[t] * x[t];
        h[t] = h_t;
        prev_h = h_t;
    }
}

struct ScanTest : public ProblemTest<1> {
    std::vector<float> h_A, h_B, h_x, h_h, h_h_ref;
    float *d_A = nullptr, *d_B = nullptr, *d_x = nullptr, *d_h = nullptr;

    ScanTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int N = size.dims[0];
        h_A.resize(N); h_B.resize(N); h_x.resize(N);
        h_h.assign(N, 0.0f); h_h_ref.assign(N, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist_A(0.9f, 1.1f);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < N; ++i) {
            h_A[i] = dist_A(gen);
            h_B[i] = dist(gen);
            h_x[i] = dist(gen);
        }
        if (check) cpu_scan(h_A.data(), h_B.data(), h_x.data(), h_h_ref.data(), N);
    }

    void setup_memory() {
        int N = size.dims[0];
        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, N * sizeof(float));
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_h, N * sizeof(float));
        cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_h, 0, N * sizeof(float));
    }

    void free_memory() {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_x); cudaFree(d_h);
    }

    void setup_reference() override { setup_memory(); }
    void teardown_reference() override { free_memory(); }
    void setup_student() override { setup_memory(); }
    void teardown_student() override { free_memory(); }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_parallel_associative_scan(d_A, d_B, d_x, d_h, size.dims[0]);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_parallel_associative_scan(d_A, d_B, d_x, d_h, size.dims[0]);
    }

    CorrectnessResult verify() override {
        int N = size.dims[0];
        cudaMemcpy(h_h.data(), d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_h_ref.data(), h_h.data(), N, 1e-3f);
    }

    void print_mismatch() override {
        print_array("Expected Output", h_h_ref.data(), size.dims[0]);
        print_array("Actual Output", h_h.data(), size.dims[0]);
    }

    double get_bandwidth_bytes() override {
        return size.dims[0] * 4 * sizeof(float);
    }
    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);
    std::vector<TestSize<1>> correctness_sizes = {{128}, {256}, {512}, {1024}};
    std::vector<TestSize<1>> perf_sizes = {{1024}}; // Keep perf small because reference handles only 1 block
    run_test_suite<1, ScanTest>("Exercise 38: Parallel Associative Scan", config, correctness_sizes, perf_sizes);
    return 0;
}
