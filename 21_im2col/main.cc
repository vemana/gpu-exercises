

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

void cpu_baseline(const float* input, float* output, long long C, long long H, long long W) {
    for (long long c = 0; c < C; ++c) {
        for (long long y = 0; y < H; ++y) {
            for (long long x = 0; x < W; ++x) {
                long long col_col = y * W + x;
                for (int fy = -1; fy <= 1; ++fy) {
                    for (int fx = -1; fx <= 1; ++fx) {
                        long long col_row = c * 9 + (fy + 1) * 3 + (fx + 1);
                        long long in_y = y + fy;
                        long long in_x = x + fx;
                        
                        float val = 0.0f;
                        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                            val = input[c * H * W + in_y * W + in_x];
                        }
                        output[col_row * (H * W) + col_col] = val;
                    }
                }
            }
        }
    }
}

struct Im2ColTest : public ProblemTest<3> {
    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_output_ref;

    float *d_input = nullptr;
    float *d_output = nullptr;

    Im2ColTest(const TestSize<3>& size) : ProblemTest<3>(size) {}

    void generate_test_data(bool check) override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        
        long long n_in = c * h * w;
        long long n_out = c * 9 * h * w;
        
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
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        long long n_in = c * h * w;
        long long n_out = c * 9 * h * w;
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_output, n_out * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_out * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_im2col(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_reference() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    void setup_student() override {
        long long c = size.dims[0];
        long long h = size.dims[1];
        long long w = size.dims[2];
        long long n_in = c * h * w;
        long long n_out = c * 9 * h * w;
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMalloc(&d_output, n_out * sizeof(float));
        cudaMemcpy(d_input, h_input.data(), n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_out * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_im2col(d_input, d_output, size.dims[0], size.dims[1], size.dims[2]);
    }

    void teardown_student() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }

    CorrectnessResult verify() override {
        long long n_out = size.dims[0] * 9 * size.dims[1] * size.dims[2];
        cudaMemcpy(h_output.data(), d_output, n_out * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_output_ref.data(), h_output.data(), n_out, 1e-4f);
    }

    void print_mismatch() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * 9 * size.dims[1] * size.dims[2];
        print_array("Input", h_input.data(), n_in);
        print_array("Expected Output", h_output_ref.data(), n_out);
        print_array("Actual Output", h_output.data(), n_out);
    }

    double get_bandwidth_bytes() override {
        long long n_in = size.dims[0] * size.dims[1] * size.dims[2];
        long long n_out = size.dims[0] * 9 * size.dims[1] * size.dims[2];
        return (n_in + n_out) * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<3> config = parse_args<3>(argc, argv);

    std::vector<TestSize<3>> correctness_sizes = {
        {1, 3, 3}, {2, 5, 5}, {3, 16, 16}, {8, 32, 32}
    };
    
    std::vector<TestSize<3>> perf_sizes = {
        {16, 128, 128}, {32, 256, 256}
    };

    run_test_suite<3, Im2ColTest>("Exercise 21: Im2Col", config, correctness_sizes, perf_sizes);
    return 0;
}
