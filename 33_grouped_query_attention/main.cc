
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

void cpu_gqa(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    int group_ratio = num_q_heads / num_kv_heads;
    
    for (int q_idx = 0; q_idx < num_q_heads; ++q_idx) {
        int kv_idx = q_idx / group_ratio;
        
        for (int i = 0; i < seq_len; ++i) {
            const float* my_q = &Q[(q_idx * seq_len + i) * head_dim];
            
            std::vector<float> scores(seq_len, 0.0f);
            float max_score = -1e20f;
            
            for (int j = 0; j < seq_len; ++j) {
                const float* my_k = &K[(kv_idx * seq_len + j) * head_dim];
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += my_q[d] * my_k[d];
                }
                score *= scale;
                scores[j] = score;
                max_score = std::max(max_score, score);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            
            float* my_out = &Out[(q_idx * seq_len + i) * head_dim];
            for (int d = 0; d < head_dim; ++d) my_out[d] = 0.0f;
            
            for (int j = 0; j < seq_len; ++j) {
                float prob = scores[j] / sum_exp;
                const float* my_v = &V[(kv_idx * seq_len + j) * head_dim];
                for (int d = 0; d < head_dim; ++d) {
                    my_out[d] += prob * my_v[d];
                }
            }
        }
    }
}

struct GQATest : public ProblemTest<1> {
    int seq_len;
    int num_q_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 64;
    
    std::vector<float> h_Q, h_K, h_V, h_Out, h_Out_ref;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Out = nullptr;

    GQATest(const TestSize<1>& size) : ProblemTest<1>(size) {
        seq_len = size.dims[0];
    }

    void generate_test_data(bool check) override {
        h_Q.resize(num_q_heads * seq_len * head_dim);
        h_K.resize(num_kv_heads * seq_len * head_dim);
        h_V.resize(num_kv_heads * seq_len * head_dim);
        h_Out.resize(num_q_heads * seq_len * head_dim, 0.0f);
        h_Out_ref.resize(num_q_heads * seq_len * head_dim, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (float& val : h_Q) val = dist(gen);
        for (float& val : h_K) val = dist(gen);
        for (float& val : h_V) val = dist(gen);

        if (check) {
            cpu_gqa(h_Q, h_K, h_V, h_Out_ref, seq_len, num_q_heads, num_kv_heads, head_dim);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_Q, h_Q.size() * sizeof(float));
        cudaMalloc(&d_K, h_K.size() * sizeof(float));
        cudaMalloc(&d_V, h_V.size() * sizeof(float));
        cudaMalloc(&d_Out, h_Out.size() * sizeof(float));
        
        cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_gqa_reference(d_Q, d_K, d_V, d_Out, seq_len, num_q_heads, num_kv_heads, head_dim);
    }

    void teardown_reference() override {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out);
    }

    void setup_student() override {
        cudaMalloc(&d_Q, h_Q.size() * sizeof(float));
        cudaMalloc(&d_K, h_K.size() * sizeof(float));
        cudaMalloc(&d_V, h_V.size() * sizeof(float));
        cudaMalloc(&d_Out, h_Out.size() * sizeof(float));
        
        cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_gqa(d_Q, d_K, d_V, d_Out, seq_len, num_q_heads, num_kv_heads, head_dim);
    }

    void teardown_student() override {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Out);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_Out.data(), d_Out, h_Out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < h_Out.size(); ++i) {
            float diff = std::abs(h_Out_ref[i] - h_Out[i]);
            float magnitude = std::abs(h_Out_ref[i]) + 1e-5f;
            if (diff > 1e-3f && (diff / magnitude) > 1e-2f) {
                return {false, i, h_Out_ref[i], h_Out[i]};
            }
        }
        return {true, -1, 0.0f, 0.0f};
    }

    void print_mismatch() override {
        std::cout << "--- Expected vs Actual ---" << std::endl;
        for (int i = 0; i < std::min((int)h_Out.size(), 20); ++i) {
            std::cout << "Idx " << i << ": Expected " << h_Out_ref[i] << " | Actual " << h_Out[i] << std::endl;
        }
    }

    double get_bandwidth_bytes() override {
        double bytes = 0;
        // Read Q: 1 time
        bytes += num_q_heads * seq_len * head_dim * sizeof(float);
        // Read K and V: For each token, thread reads all K and V of corresponding KV head
        // Since threads run independently, they redundantly load K and V.
        bytes += num_q_heads * seq_len * seq_len * head_dim * sizeof(float) * 2;
        // Write Out
        bytes += num_q_heads * seq_len * head_dim * sizeof(float);
        return bytes;
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Sequence Length
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{16}}, {{256}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{512}}, {{1024}}
    };

    run_test_suite<1, GQATest>("Exercise 33: Grouped-Query Attention", config, correctness_sizes, perf_sizes);
    return 0;
}
