

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "kernel.h"
#include "reference_kernel.h"

void cpu_w8a16_linear(
    const std::vector<float>& X, const std::vector<int8_t>& W_int8, const std::vector<float>& scales, std::vector<float>& Y,
    int M, int N, int K, int group_size) {
    
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int group_id = k / group_size;
                float scale = scales[n * (K / group_size) + group_id];
                float w_float = (float)W_int8[n * K + k] * scale;
                sum += X[m * K + k] * w_float;
            }
            Y[m * N + n] = sum;
        }
    }
}

struct W8A16LinearTest : public ProblemTest<1> {
    int M;
    int N = 1024;
    int K = 4096;
    int group_size = 128;
    
    std::vector<float> h_X, h_scales, h_Y, h_Y_ref;
    std::vector<int8_t> h_W_int8;

    float *d_X = nullptr, *d_scales = nullptr, *d_Y = nullptr;
    int8_t *d_W_int8 = nullptr;

    W8A16LinearTest(const TestSize<1>& size) : ProblemTest<1>(size) {
        M = size.dims[0];
    }

    void generate_test_data(bool check) override {
        h_X.resize(M * K);
        h_W_int8.resize(N * K);
        h_scales.resize(N * (K / group_size));
        h_Y.resize(M * N, 0.0f);
        h_Y_ref.resize(M * N, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> i_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.05f);
        
        for (float& val : h_X) val = f_dist(gen);
        for (int8_t& val : h_W_int8) val = (int8_t)i_dist(gen);
        for (float& val : h_scales) val = scale_dist(gen);

        if (check) {
            cpu_w8a16_linear(h_X, h_W_int8, h_scales, h_Y_ref, M, N, K, group_size);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_X, h_X.size() * sizeof(float));
        cudaMalloc(&d_W_int8, h_W_int8.size() * sizeof(int8_t));
        cudaMalloc(&d_scales, h_scales.size() * sizeof(float));
        cudaMalloc(&d_Y, h_Y.size() * sizeof(float));
        
        cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_int8, h_W_int8.data(), h_W_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_w8a16_linear_reference(d_X, d_W_int8, d_scales, d_Y, M, N, K, group_size);
    }

    void teardown_reference() override {
        cudaFree(d_X); cudaFree(d_W_int8); cudaFree(d_scales); cudaFree(d_Y);
    }

    void setup_student() override {
        cudaMalloc(&d_X, h_X.size() * sizeof(float));
        cudaMalloc(&d_W_int8, h_W_int8.size() * sizeof(int8_t));
        cudaMalloc(&d_scales, h_scales.size() * sizeof(float));
        cudaMalloc(&d_Y, h_Y.size() * sizeof(float));
        
        cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_int8, h_W_int8.data(), h_W_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_w8a16_linear(d_X, d_W_int8, d_scales, d_Y, M, N, K, group_size);
    }

    void teardown_student() override {
        cudaFree(d_X); cudaFree(d_W_int8); cudaFree(d_scales); cudaFree(d_Y);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_Y.data(), d_Y, h_Y.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < h_Y.size(); ++i) {
            float diff = std::abs(h_Y_ref[i] - h_Y[i]);
            float magnitude = std::abs(h_Y_ref[i]) + 1e-5f;
            if (diff > 1e-2f && (diff / magnitude) > 1e-2f) {
                return {false, i, h_Y_ref[i], h_Y[i]};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min((int)h_Y.size(), 20); ++i) {
            std::cout << "Idx " << i << ": Expected " << h_Y_ref[i] << " | Actual " << h_Y[i] << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        // Read X, W_int8, scales, Write Y
        return M * K * sizeof(float) + N * K * sizeof(int8_t) + N * (K / group_size) * sizeof(float) + M * N * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // M: Batch size (number of tokens)
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{4}}, {{16}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{32}}, {{64}}
    };

    run_test_suite<1, W8A16LinearTest>("Exercise 35: W8A16 Linear (Weight-Only Quantization)", config, correctness_sizes, perf_sizes);
    return 0;
}
