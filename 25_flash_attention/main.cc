

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

void cpu_baseline(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D) {
    float scale = 1.0f / std::sqrt((float)D);
    
    for (long long b = 0; b < B; ++b) {
        for (long long n = 0; n < N; ++n) {
            long long head_offset = (b * N + n) * S * D;
            
            for (long long s_q = 0; s_q < S; ++s_q) {
                // Find max score
                float max_val = -1e20f;
                std::vector<float> scores(S, 0.0f);
                
                for (long long s_k = 0; s_k < S; ++s_k) {
                    float score = 0.0f;
                    for (long long d = 0; d < D; ++d) {
                        score += q[head_offset + s_q * D + d] * k[head_offset + s_k * D + d];
                    }
                    score *= scale;
                    scores[s_k] = score;
                    if (score > max_val) max_val = score;
                }
                
                // Softmax and output calculation
                float sum_exp = 0.0f;
                for (long long s_k = 0; s_k < S; ++s_k) {
                    scores[s_k] = std::exp(scores[s_k] - max_val);
                    sum_exp += scores[s_k];
                }
                
                float inv_sum = 1.0f / sum_exp;
                for (long long d = 0; d < D; ++d) {
                    float out_val = 0.0f;
                    for (long long s_k = 0; s_k < S; ++s_k) {
                        float prob = scores[s_k] * inv_sum;
                        out_val += prob * v[head_offset + s_k * D + d];
                    }
                    o[head_offset + s_q * D + d] = out_val;
                }
            }
        }
    }
}

struct FlashAttentionTest : public ProblemTest<4> {
    std::vector<float> h_q;
    std::vector<float> h_k;
    std::vector<float> h_v;
    std::vector<float> h_o;
    std::vector<float> h_o_ref;

    float *d_q = nullptr;
    float *d_k = nullptr;
    float *d_v = nullptr;
    float *d_o = nullptr;

    FlashAttentionTest(const TestSize<4>& size) : ProblemTest<4>(size) {}

    void generate_test_data(bool check) override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        h_q.resize(num_elements);
        h_k.resize(num_elements);
        h_v.resize(num_elements);
        h_o.assign(num_elements, 0.0f);
        h_o_ref.assign(num_elements, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (long long i = 0; i < num_elements; ++i) {
            h_q[i] = dist(gen);
            h_k[i] = dist(gen);
            h_v[i] = dist(gen);
        }
        if (check) cpu_baseline(h_q.data(), h_k.data(), h_v.data(), h_o_ref.data(), size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void setup_reference() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMalloc(&d_q, num_elements * sizeof(float));
        cudaMalloc(&d_k, num_elements * sizeof(float));
        cudaMalloc(&d_v, num_elements * sizeof(float));
        cudaMalloc(&d_o, num_elements * sizeof(float));
        cudaMemcpy(d_q, h_q.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_o, 0, num_elements * sizeof(float));
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_reference_flash_attention(d_q, d_k, d_v, d_o, size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void teardown_reference() override {
        if (d_q) cudaFree(d_q);
        if (d_k) cudaFree(d_k);
        if (d_v) cudaFree(d_v);
        if (d_o) cudaFree(d_o);
    }

    void setup_student() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMalloc(&d_q, num_elements * sizeof(float));
        cudaMalloc(&d_k, num_elements * sizeof(float));
        cudaMalloc(&d_v, num_elements * sizeof(float));
        cudaMalloc(&d_o, num_elements * sizeof(float));
        cudaMemcpy(d_q, h_q.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_o, 0, num_elements * sizeof(float));
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_flash_attention(d_q, d_k, d_v, d_o, size.dims[0], size.dims[1], size.dims[2], size.dims[3]);
    }

    void teardown_student() override {
        if (d_q) cudaFree(d_q);
        if (d_k) cudaFree(d_k);
        if (d_v) cudaFree(d_v);
        if (d_o) cudaFree(d_o);
    }

    CorrectnessResult verify() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        cudaMemcpy(h_o.data(), d_o, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        return check_correctness(h_o_ref.data(), h_o.data(), num_elements, 1e-3f);
    }

    void print_mismatch() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        print_array("Q", h_q.data(), num_elements);
        print_array("K", h_k.data(), num_elements);
        print_array("V", h_v.data(), num_elements);
        print_array("Expected Output", h_o_ref.data(), num_elements);
        print_array("Actual Output", h_o.data(), num_elements);
    }

    double get_bandwidth_bytes() override {
        long long num_elements = size.dims[0] * size.dims[1] * size.dims[2] * size.dims[3];
        return 4.0 * num_elements * sizeof(float);
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<4> config = parse_args<4>(argc, argv);

    std::vector<TestSize<4>> correctness_sizes = {
        {1, 1, 16, 16}, {1, 2, 64, 32}, {2, 4, 128, 64}
    };
    
    std::vector<TestSize<4>> perf_sizes = {
        {2, 8, 1024, 64}, {4, 12, 2048, 64}
    };

    run_test_suite<4, FlashAttentionTest>("Exercise 25: Flash Attention", config, correctness_sizes, perf_sizes);
    return 0;
}
