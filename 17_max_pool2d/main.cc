

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

void cpu_baseline(const float* input, float* output, long long channels, long long height, long long width) {
    long long out_height = height / 2;
    long long out_width = width / 2;
    for (long long c = 0; c < channels; ++c) {
        for (long long y = 0; y < out_height; ++y) {
            for (long long x = 0; x < out_width; ++x) {
                long long in_y = y * 2;
                long long in_x = x * 2;
                long long channel_offset = c * height * width;
                
                float val0 = input[channel_offset + in_y * width + in_x];
                float val1 = input[channel_offset + in_y * width + in_x + 1];
                float val2 = input[channel_offset + (in_y + 1) * width + in_x];
                float val3 = input[channel_offset + (in_y + 1) * width + in_x + 1];
                
                float max_val = std::max({val0, val1, val2, val3});
                output[(c * out_height + y) * out_width + x] = max_val;
            }
        }
    }
}

struct MaxPool2DTest : public ProblemTest<3> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    MaxPool2DTest(const TestSize<3>& size) : ProblemTest<3>(size) {}

    void generate_test_data(bool check) override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        
        long long n_in = c * h * w;
        long long n_out = c * (h / 2) * (w / 2);
        
        h_input.resize(n_in);
        h_output.assign(n_out, 0.0f);
        h_output_ref.assign(n_out, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (long long i = 0; i < n_in; ++i) {
            h_input[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_output_ref.data(), c, h, w);
    }

    void setup_reference() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * (size.dims[1] / 2) * (size.dims[2] / 2);
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_output, n_out * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_out * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_max_pool2d(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * (size.dims[1] / 2) * (size.dims[2] / 2);
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_output, n_out * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_out * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_max_pool2d(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n_out = size.dims[0] * (size.dims[1] / 2) * (size.dims[2] / 2);
        cudaMemcpy(h_output.data(), d_output, n_out * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n_out, 1e-4f);
    }

    void print_mismatch() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * (size.dims[1] / 2) * (size.dims[2] / 2);
        print_array("Input", h_input.data(), n_in);
        print_array("Expected Output", h_output_ref.data(), n_out);
        print_array("Actual Output", h_output.data(), n_out);
    }

    double get_bandwidth_bytes() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * (size.dims[1] / 2) * (size.dims[2] / 2);
        return (n_in + n_out) * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<3> config = parse_args<3>(argc, argv);

    // Keep H and W even. C, H, W
    std::vector<TestSize<3>> correctness_sizes = {
        {1, 2, 2}, {3, 4, 4}, {3, 30, 30}, {3, 32, 32}, {4, 64, 64}, {8, 128, 128}
    };
    
    std::vector<TestSize<3>> perf_sizes = {
        {16, 512, 512}, {32, 1024, 1024}, {64, 1024, 1024}
    };

    run_test_suite<3, MaxPool2DTest>("Exercise 17: Max Pooling 2D", config, correctness_sizes, perf_sizes);
    return 0;
}
