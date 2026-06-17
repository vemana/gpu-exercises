

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

void cpu_baseline(const float* input, float* output, long long batch, long long seq_len, long long hidden_dim) {
    for (long long b = 0; b < batch; ++b) {
        for (long long s = 0; s < seq_len; ++s) {
            for (long long h = 0; h < hidden_dim / 2; ++h) {
                float freq = std::pow(10000.0f, -2.0f * (float)h / (float)hidden_dim);
                float theta = (float)s * freq;
                float cos_val = std::cos(theta);
                float sin_val = std::sin(theta);
                
                long long idx1 = b * seq_len * hidden_dim + s * hidden_dim + 2 * h;
                long long idx2 = b * seq_len * hidden_dim + s * hidden_dim + 2 * h + 1;
                
                float x1 = input[idx1];
                float x2 = input[idx2];
                
                output[idx1] = x1 * cos_val - x2 * sin_val;
                output[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

struct RoPETest : public ProblemTest<3> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    RoPETest(const TestSize<3>& size) : ProblemTest<3>(size) {}

    void generate_test_data(bool check) override {
        long long batch = size.dims[0];
        long long seq_len = size.dims[1];
        long long hidden_dim = size.dims[2];
        long long n = batch * seq_len * hidden_dim;
        
        h_input.resize(n);
        h_output.assign(n, 0.0f);
        h_output_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (long long i = 0; i < n; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), batch, seq_len, hidden_dim);
    }

    void setup_reference() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_rope(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_rope(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n, 1e-4f);
    }

    void print_mismatch() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        print_array("Input", h_input.data(), n);
        print_array("Expected Output", h_output_ref.data(), n);
        print_array("Actual Output", h_output.data(), n);
    }

    double get_bandwidth_bytes() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        return 2.0 * n * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<3> config = parse_args<3>(argc, argv);

    // Keep hidden_dim even
    std::vector<TestSize<3>> correctness_sizes = {
        {1, 3, 2}, {2, 5, 4}, {3, 16, 64}, {8, 32, 128}
    };
    
    std::vector<TestSize<3>> perf_sizes = {
        {16, 512, 256}, {32, 1024, 256}
    };

    run_test_suite<3, RoPETest>("Exercise 24: RoPE", config, correctness_sizes, perf_sizes);
    return 0;
}
