

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

void cpu_baseline(const float* input, float* output, long long N, long long C, long long H, long long W) {
    long long spatial_size = H * W;
    long long elements_per_channel = N * spatial_size;

    for (long long c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                sum += input[(n * C + c) * spatial_size + s];
            }
        }
        float mean = sum / elements_per_channel;

        float var_sum = 0.0f;
        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                float diff = input[(n * C + c) * spatial_size + s] - mean;
                var_sum += diff * diff;
            }
        }
        float var = var_sum / elements_per_channel;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                long long idx = (n * C + c) * spatial_size + s;
                output[idx] = (input[idx] - mean) * inv_std;
            }
        }
    }
}

struct BatchNormTest : public ProblemTest<4> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    BatchNormTest(const TestSize<4>& size) : ProblemTest<4>(size) {}

    void generate_test_data(bool check) override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        h_input.resize(num_elements);
        h_output.assign(num_elements, 0.0f);
        h_output_ref.assign(num_elements, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (long long i = 0; i < num_elements; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void setup_reference() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMalloc(&d_input, num_elements * sizeof(float));
        cudaMalloc(&d_output, num_elements * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, num_elements * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_batch_norm(d_input, d_output, size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMalloc(&d_input, num_elements * sizeof(float));
        cudaMalloc(&d_output, num_elements * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, num_elements * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_batch_norm(d_input, d_output, size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), num_elements, 1e-3f);
    }

    void print_mismatch() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        print_array("Input", h_input.data(), num_elements);
        print_array("Expected Output", h_output_ref.data(), num_elements);
        print_array("Actual Output", h_output.data(), num_elements);
    }

    double get_bandwidth_bytes() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        return 2.0 * num_elements * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<4> config = parse_args<4>(argc, argv);

    std::vector<TestSize<4>> correctness_sizes = {
        {1, 3, 2, 2}, {2, 3, 4, 4}, {4, 16, 16, 16}, {8, 32, 32, 32}
    };
    
    std::vector<TestSize<4>> perf_sizes = {
        {32, 64, 64, 64}, {64, 64, 128, 128}
    };

    run_test_suite<4, BatchNormTest>("Exercise 19: Batch Normalization", config, correctness_sizes, perf_sizes);
    return 0;
}
