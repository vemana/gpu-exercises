#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "../utils/framework.h"
#include "../utils/argparse.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_baseline(const float* input, const float* filter, float* output, long long C, long long H, long long W) {
    for (long long c = 0; c < C; ++c) {
        for (long long y = 0; y < H; ++y) {
            for (long long x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int fy = -1; fy <= 1; ++fy) {
                    for (int fx = -1; fx <= 1; ++fx) {
                        long long in_y = y + fy;
                        long long in_x = x + fx;
                        
                        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                            float pixel = input[c * H * W + in_y * W + in_x];
                            float weight = filter[c * 9 + (fy + 1) * 3 + (fx + 1)];
                            sum += pixel * weight;
                        }
                    }
                }
                output[c * H * W + y * W + x] = sum;
            }
        }
    }
}

struct DepthwiseConv2DTest : public ProblemTest<3> {
    std::vector<float> h_input;
    std::vector<float> h_filter;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;

    DepthwiseConv2DTest(const TestSize<3>& size) : ProblemTest<3>(size) {}

    void generate_test_data(bool check) override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        
        long long n_in = c * h * w;
        long long n_filter = c * 9;
        
        h_input.resize(n_in);
        h_filter.resize(n_filter);
        h_output.assign(n_in, 0.0f);
        h_output_ref.assign(n_in, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (long long i = 0; i < n_in; ++i) {
            h_input[i] = dist(gen);
        }
        for (long long i = 0; i < n_filter; ++i) {
            h_filter[i] = dist(gen);
        }
        if (check) cpu_baseline(h_input.data(), h_filter.data(), h_output_ref.data(), c, h, w);
    }

    void setup_reference() override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        long long n_in = c * h * w;
        long long n_filter = c * 9;
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_filter, n_filter * sizeof(float));
        cudaMalloc(&d_output, n_in * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter.data(), n_filter * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_in * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_depthwise_conv2d(d_input, d_filter, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_filter) cudaFree(d_filter);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        long long n_in = c * h * w;
        long long n_filter = c * 9;
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_filter, n_filter * sizeof(float));
        cudaMalloc(&d_output, n_in * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter.data(), n_filter * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_in * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_depthwise_conv2d(d_input, d_filter, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_filter) cudaFree(d_filter);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n = size.dims[0] * size.dims[1] * size.dims[2];
        cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n, 1e-3f);
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

    std::vector<TestSize<3>> correctness_sizes = {
        {1, 3, 3}, {3, 15, 15}, {4, 16, 16}, {8, 31, 31}, {16, 32, 32}
    };
    
    std::vector<TestSize<3>> perf_sizes = {
        {16, 512, 512}, {32, 1024, 1024}
    };

    run_test_suite<3, DepthwiseConv2DTest>("Exercise 22: Depthwise Conv2D", config, correctness_sizes, perf_sizes);
    return 0;
}
