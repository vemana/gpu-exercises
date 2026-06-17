#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "../utils/utils.h"
#include "kernel.h"
#include "reference_kernel.h"

// CPU Baseline: Computes attention but restricts the inner loop to start_j = max(0, i - W + 1).
// Emulates the exact banded sparse attention matrix.
void cpu_sliding_window_attention(const float* Q, const float* K, const float* V, float* O, int S, int D, int W) {
    for (int i = 0; i < S; ++i) {
        float max_score = -1e20f;
        int start_j = std::max(0, i - W + 1);
        int end_j = i;
        
        for (int j = start_j; j <= end_j; ++j) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q[i * D + d] * K[j * D + d];
            }
            score /= std::sqrt((float)D);
            if (score > max_score) max_score = score;
        }
        
        float sum_exp = 0.0f;
        for (int j = start_j; j <= end_j; ++j) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q[i * D + d] * K[j * D + d];
            }
            score /= std::sqrt((float)D);
            sum_exp += std::exp(score - max_score);
        }
        
        for (int d = 0; d < D; ++d) {
            float out_val = 0.0f;
            for (int j = start_j; j <= end_j; ++j) {
                float score = 0.0f;
                for (int d_in = 0; d_in < D; ++d_in) {
                    score += Q[i * D + d_in] * K[j * D + d_in];
                }
                score /= std::sqrt((float)D);
                float prob = std::exp(score - max_score) / sum_exp;
                out_val += prob * V[j * D + d];
            }
            O[i * D + d] = out_val;
        }
    }
}

struct SWATest : public ProblemTest<1> {
    int D = 64;
    int W = 32;

    std::vector<float> h_Q, h_K, h_V, h_O, h_O_ref;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;

    SWATest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int S = size.dims[0];
        h_Q.resize(S * D); h_K.resize(S * D); h_V.resize(S * D);
        h_O.assign(S * D, 0.0f); h_O_ref.assign(S * D, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < S * D; ++i) {
            h_Q[i] = dist(gen); h_K[i] = dist(gen); h_V[i] = dist(gen);
        }
        if (check) cpu_sliding_window_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), S, D, W);
    }

    void setup_memory() {
        int S = size.dims[0];
        cudaMalloc(&d_Q, S * D * sizeof(float));
        cudaMalloc(&d_K, S * D * sizeof(float));
        cudaMalloc(&d_V, S * D * sizeof(float));
        cudaMalloc(&d_O, S * D * sizeof(float));
        cudaMemcpy(d_Q, h_Q.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_O, 0, S * D * sizeof(float));
    }

    void free_memory() { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); }

    void setup_reference() override { setup_memory(); }
    void teardown_reference() override { free_memory(); }
    void setup_student() override { setup_memory(); }
    void teardown_student() override { free_memory(); }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_sliding_window_attention(d_Q, d_K, d_V, d_O, size.dims[0], D, W);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_sliding_window_attention(d_Q, d_K, d_V, d_O, size.dims[0], D, W);
    }

    CorrectnessResult verify() override {
        int S = size.dims[0];
        cudaMemcpy(h_O.data(), d_O, S * D * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_O_ref.data(), h_O.data(), S * D, 1e-3f);
    }

    void print_mismatch() override {
        print_array("Expected Output", h_O_ref.data(), size.dims[0] * D);
        print_array("Actual Output", h_O.data(), size.dims[0] * D);
    }

    double get_bandwidth_bytes() override {
        return size.dims[0] * D * 4 * sizeof(float);
    }
    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);
    std::vector<TestSize<1>> correctness_sizes = {{128}, {256}, {512}};
    std::vector<TestSize<1>> perf_sizes = {{1024}, {2048}, {4096}};
    run_test_suite<1, SWATest>("Exercise 39: Sliding Window Attention", config, correctness_sizes, perf_sizes);
    return 0;
}
