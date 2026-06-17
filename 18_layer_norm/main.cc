

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

void cpu_baseline(const float* input, float* output, long long batch_seq, long long hidden_dim) {
    for (long long b = 0; b < batch_seq; ++b) {
        float sum = 0.0f;
        for (long long h = 0; h < hidden_dim; ++h) {
            sum += input[b * hidden_dim + h];
        }
        float mean = sum / hidden_dim;
        
        float var_sum = 0.0f;
        for (long long h = 0; h < hidden_dim; ++h) {
            float diff = input[b * hidden_dim + h] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden_dim;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        
        for (long long h = 0; h < hidden_dim; ++h) {
            output[b * hidden_dim + h] = (input[b * hidden_dim + h] - mean) * inv_std;
        }
    }
}

struct LayerNormTest : public ProblemTest<2> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    LayerNormTest(const TestSize<2>& size) : ProblemTest<2>(size) {}

    void generate_test_data(bool check) override {
        long long batch_seq = size.dims[0];
        long long hidden_dim = size.dims[1];
        long long n = batch_seq * hidden_dim;
        
        h_input.resize(n);
        h_output.assign(n, 0.0f);
        h_output_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (long long i = 0; i < n; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), batch_seq, hidden_dim);
    }

    void setup_reference() override {
        long long n = size.dims[0] * size.dims[1];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_layer_norm(d_input, d_output, size.dims[0], size.dims[1]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long n = size.dims[0] * size.dims[1];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_layer_norm(d_input, d_output, size.dims[0], size.dims[1]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0] * size.dims[1];
        cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n, 1e-3f);
    }

    void print_mismatch() override {
        long long n = size.dims[0] * size.dims[1];
        print_array("Input", h_input.data(), n);
        print_array("Expected Output", h_output_ref.data(), n);
        print_array("Actual Output", h_output.data(), n);
    }

    double get_bandwidth_bytes() override {
        long long n = size.dims[0] * size.dims[1];
        return 2.0 * n * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<2> config = parse_args<2>(argc, argv);

    std::vector<TestSize<2>> correctness_sizes = {
        {1, 32}, {2, 64}, {3, 128}, {4, 768}, {8, 1024}, {31, 2048}
    };
    
    std::vector<TestSize<2>> perf_sizes = {
        {1024, 768}, {2048, 1024}, {4096, 2048}
    };

    run_test_suite<2, LayerNormTest>("Exercise 18: Layer Normalization", config, correctness_sizes, perf_sizes);
    return 0;
}
