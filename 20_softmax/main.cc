#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* input, float* output, long long batch_seq, long long vocab_size) {
    for (long long b = 0; b < batch_seq; ++b) {
        float max_val = -1e20f;
        for (long long v = 0; v < vocab_size; ++v) {
            if (input[b * vocab_size + v] > max_val) {
                max_val = input[b * vocab_size + v];
            }
        }
        
        float sum_exp = 0.0f;
        for (long long v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(input[b * vocab_size + v] - max_val);
        }
        
        float inv_sum = 1.0f / sum_exp;
        for (long long v = 0; v < vocab_size; ++v) {
            output[b * vocab_size + v] = std::exp(input[b * vocab_size + v] - max_val) * inv_sum;
        }
    }
}

struct SoftmaxTest : public ProblemTest<2> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    SoftmaxTest(const TestSize<2>& size) : ProblemTest<2>(size) {}

    void generate_test_data(bool check) override {
        long long batch_seq = size.dims[0];
        long long vocab_size = size.dims[1];
        long long n = batch_seq * vocab_size;
        
        h_input.resize(n);
        h_output.assign(n, 0.0f);
        h_output_ref.assign(n, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (long long i = 0; i < n; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), batch_seq, vocab_size);
    }

    void setup_reference() override {
        long long n = size.dims[0] * size.dims[1];
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_softmax(d_input, d_output, size.dims[0], size.dims[1]);
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
        return launch_softmax(d_input, d_output, size.dims[0], size.dims[1]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0] * size.dims[1];
        cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n, 1e-4f);
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
        {1, 32}, {2, 64}, {3, 1000}, {4, 1024}, {8, 2048}, {31, 4096}
    };
    
    std::vector<TestSize<2>> perf_sizes = {
        {1024, 1024}, {2048, 4096}, {4096, 8192}
    };

    run_test_suite<2, SoftmaxTest>("Exercise 20: Softmax", config, correctness_sizes, perf_sizes);
    return 0;
}
