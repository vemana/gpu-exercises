

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* input, float* output, long long size) {
    for (long long i = 0; i < size; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

struct ActivationTest : public ProblemTest<1> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    ActivationTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        long long n = size.dims[0];
        h_input.resize(n);
        h_output.assign(n, 0.0f);
        h_output_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (long long i = 0; i < n; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), n);
    }

    void setup_reference() override {
        long long n = size.dims[0];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_activation(d_input, d_output, size.dims[0]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long n = size.dims[0];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_activation(d_input, d_output, size.dims[0]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_output.data(), d_output, size.dims[0] * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), size.dims[0], 1e-4f);
    }

    void print_mismatch() override {
        print_array("Input", h_input.data(), size.dims[0]);
        print_array("Expected Output", h_output_ref.data(), size.dims[0]);
        print_array("Actual Output", h_output.data(), size.dims[0]);
    }

    double get_bandwidth_bytes() override {
        return 2.0 * size.dims[0] * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    std::vector<TestSize<1>> correctness_sizes = {
        {1}, {31}, {32}, {33}, {63}, {64}, {65}, {1023}, {1024}, {1025}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1 << 20}}, {{1 << 24}}, {{1 << 26}}, {{1 << 30}}
    };

    run_test_suite<1, ActivationTest>("Exercise 16: SiLU Activation", config, correctness_sizes, perf_sizes);
    return 0;
}
