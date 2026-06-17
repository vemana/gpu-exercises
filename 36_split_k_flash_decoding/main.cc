#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/argparse.h"
#include "../utils/framework.h"
#include "../utils/tracer.h"
#include "../utils/utils.h"
#include "kernel.h"
#include "reference_kernel.h"

// CPU Baseline for Attention
// Computes standard O(N^2) attention without split-K, serving as the ground truth.
void cpu_attention(const float* q, const float* k, const float* v, float* o, int S, int D) {
    float max_score = -1e20f;
    std::vector<float> scores(S);
    
    for (int i = 0; i < S; ++i) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += q[d] * k[i * D + d];
        }
        score /= std::sqrt((float)D);
        scores[i] = score;
        if (score > max_score) max_score = score;
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < S; ++i) {
        sum_exp += std::exp(scores[i] - max_score);
    }
    
    for (int d = 0; d < D; ++d) {
        float out_val = 0.0f;
        for (int i = 0; i < S; ++i) {
            float prob = std::exp(scores[i] - max_score) / sum_exp;
            out_val += prob * v[i * D + d];
        }
        o[d] = out_val;
    }
}

struct SplitKTest : public ProblemTest<1> {
    int D = 128;
    int split_size = 500;
    
    std::vector<float> h_q;
    std::vector<float> h_k;
    std::vector<float> h_v;
    std::vector<float> h_o;
    std::vector<float> h_o_ref;

    float *d_q = nullptr;
    float *d_k = nullptr;
    float *d_v = nullptr;
    float *d_o = nullptr;
    float *d_partial_o = nullptr;
    float *d_partial_m = nullptr;
    float *d_partial_l = nullptr;

    SplitKTest(const TestSize<1>& size) : ProblemTest<1>(size) {}

    void generate_test_data(bool check) override {
        int S = size.dims[0];
        h_q.resize(D);
        h_k.resize(S * D);
        h_v.resize(S * D);
        h_o.assign(D, 0.0f);
        h_o_ref.assign(D, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < D; ++i) h_q[i] = dist(gen);
        for (int i = 0; i < S * D; ++i) {
            h_k[i] = dist(gen);
            h_v[i] = dist(gen);
        }
        if (check) cpu_attention(h_q.data(), h_k.data(), h_v.data(), h_o_ref.data(), S, D);
    }

    void setup_memory() {
        int S = size.dims[0];
        int num_splits = (S + split_size - 1) / split_size;
        
        cudaMalloc(&d_q, D * sizeof(float));
        cudaMalloc(&d_k, S * D * sizeof(float));
        cudaMalloc(&d_v, S * D * sizeof(float));
        cudaMalloc(&d_o, D * sizeof(float));
        cudaMalloc(&d_partial_o, num_splits * D * sizeof(float));
        cudaMalloc(&d_partial_m, num_splits * sizeof(float));
        cudaMalloc(&d_partial_l, num_splits * sizeof(float));
        
        cudaMemcpy(d_q, h_q.data(), D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), S * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_o, 0, D * sizeof(float));
    }

    void free_memory() {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_partial_o);
        cudaFree(d_partial_m);
        cudaFree(d_partial_l);
    }

    void setup_reference() override { setup_memory(); }
    void teardown_reference() override { free_memory(); }
    void setup_student() override { setup_memory(); }
    void teardown_student() override { free_memory(); }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_split_k_flash_decoding(d_q, d_k, d_v, d_o, d_partial_o, d_partial_m, d_partial_l, size.dims[0], D, split_size);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_split_k_flash_decoding(d_q, d_k, d_v, d_o, d_partial_o, d_partial_m, d_partial_l, size.dims[0], D, split_size);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_o.data(), d_o, D * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_o_ref.data(), h_o.data(), D, 1e-4f);
    }

    void print_mismatch() override {
        print_array("Expected Output", h_o_ref.data(), D);
        print_array("Actual Output", h_o.data(), D);
    }

    double get_bandwidth_bytes() override {
        return (size.dims[0] * D * 2 + D * 2) * sizeof(float);
    }
    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);
    std::vector<TestSize<1>> correctness_sizes = {{1000}, {5000}, {10000}};
    std::vector<TestSize<1>> perf_sizes = {{10000}, {50000}, {100000}};
    run_test_suite<1, SplitKTest>("Exercise 36: Split-K Flash Decoding", config, correctness_sizes, perf_sizes);
    return 0;
}
