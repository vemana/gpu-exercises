

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

static unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void cpu_baseline(const float* input, float* output, long long size, float drop_prob) {
    float scale = 1.0f / (1.0f - drop_prob);
    for (long long i = 0; i < size; ++i) {
        unsigned int hash_val = pcg_hash((unsigned int)i);
        float rand_val = (float)(hash_val % 10000) / 10000.0f;
        
        if (rand_val < drop_prob) {
            output[i] = 0.0f;
        } else {
            output[i] = input[i] * scale;
        }
    }
}

struct DropoutTest : public ProblemTest<1> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    float drop_prob = 0.5f;

    DropoutTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

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
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), n, drop_prob);
    }

    void setup_reference() override {
        long long n = size.dims[0];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_dropout(d_input, d_output, size.dims[0], drop_prob);
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
        return launch_dropout(d_input, d_output, size.dims[0], drop_prob);
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
        {{1 << 20}}, {{1 << 24}}, {{1 << 26}}, {{1 << 28}}
    };

    run_test_suite<1, DropoutTest>("Exercise 23: Dropout", config, correctness_sizes, perf_sizes);
    return 0;
}
