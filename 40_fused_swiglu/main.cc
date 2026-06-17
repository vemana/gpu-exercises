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

// CPU Baseline: Element-wise SwiGLU computation.
// Math: O = (X / (1 + exp(-X))) * Gate
void cpu_swiglu(const float* X, const float* gate, float* O, int total_elements) {
    for (int i = 0; i < total_elements; ++i) {
        float x = X[i];
        float g = gate[i];
        float silu = x / (1.0f + std::exp(-x));
        O[i] = silu * g;
    }
}

struct SwiGLUTest : public ProblemTest<1> {
    std::vector<float> h_X, h_gate, h_O, h_O_ref;
    float *d_X = nullptr, *d_gate = nullptr, *d_O = nullptr;

    SwiGLUTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int N = size.dims[0];
        h_X.resize(N); h_gate.resize(N);
        h_O.assign(N, 0.0f); h_O_ref.assign(N, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (int i = 0; i < N; ++i) {
            h_X[i] = dist(gen);
            h_gate[i] = dist(gen);
        }
        if (check) cpu_swiglu(h_X.data(), h_gate.data(), h_O_ref.data(), N);
    }

    void setup_memory() {
        int N = size.dims[0];
        cudaMalloc(&d_X, N * sizeof(float));
        cudaMalloc(&d_gate, N * sizeof(float));
        cudaMalloc(&d_O, N * sizeof(float));
        cudaMemcpy(d_X, h_X.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate, h_gate.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_O, 0, N * sizeof(float));
    }

    void free_memory() { cudaFree(d_X); cudaFree(d_gate); cudaFree(d_O); }

    void setup_reference() override { setup_memory(); }
    void teardown_reference() override { free_memory(); }
    void setup_student() override { setup_memory(); }
    void teardown_student() override { free_memory(); }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_fused_swiglu(d_X, d_gate, d_O, size.dims[0]);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_fused_swiglu(d_X, d_gate, d_O, size.dims[0]);
    }

    CorrectnessResult verify() override {
        int N = size.dims[0];
        cudaMemcpy(h_O.data(), d_O, N * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_O_ref.data(), h_O.data(), N, 1e-4f);
    }

    void print_mismatch() override {
        print_array("Expected Output", h_O_ref.data(), size.dims[0]);
        print_array("Actual Output", h_O.data(), size.dims[0]);
    }

    double get_bandwidth_bytes() override {
        return size.dims[0] * 3 * sizeof(float);
    }
    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);
    std::vector<TestSize<1>> correctness_sizes = {{1024}, {4096}, {16384}};
    std::vector<TestSize<1>> perf_sizes = {{1000000}, {5000000}, {10000000}};
    run_test_suite<1, SwiGLUTest>("Exercise 40: Fused SwiGLU MLP", config, correctness_sizes, perf_sizes);
    return 0;
}
