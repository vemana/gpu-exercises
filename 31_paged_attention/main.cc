
// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

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

void cpu_paged_attention(
    const std::vector<float>& Q, const std::vector<float>& K_cache, const std::vector<float>& V_cache,
    const std::vector<int>& block_tables, const std::vector<int>& context_lens, std::vector<float>& Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq) {
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    
    for (int i = 0; i < num_seqs; ++i) {
        int ctx_len = context_lens[i];
        std::vector<float> logits(ctx_len, 0.0f);
        
        const float* q = &Q[i * head_dim];
        const int* seq_table = &block_tables[i * max_blocks_per_seq];
        
        float max_logit = -1e20f;
        
        // 1. Compute QK^T
        for (int t = 0; t < ctx_len; ++t) {
            int logical_block = t / block_size;
            int physical_block = seq_table[logical_block];
            int offset = t % block_size;
            
            const float* k = &K_cache[(physical_block * block_size + offset) * head_dim];
            
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[d] * k[d];
            }
            score *= scale;
            logits[t] = score;
            max_logit = std::max(max_logit, score);
        }
        
        // 2. Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < ctx_len; ++t) {
            logits[t] = std::exp(logits[t] - max_logit);
            sum_exp += logits[t];
        }
        
        // 3. Weighted sum of V
        float* out = &Out[i * head_dim];
        for (int d = 0; d < head_dim; ++d) {
            out[d] = 0.0f;
        }
        
        for (int t = 0; t < ctx_len; ++t) {
            float prob = logits[t] / sum_exp;
            int logical_block = t / block_size;
            int physical_block = seq_table[logical_block];
            int offset = t % block_size;
            
            const float* v = &V_cache[(physical_block * block_size + offset) * head_dim];
            
            for (int d = 0; d < head_dim; ++d) {
                out[d] += prob * v[d];
            }
        }
    }
}

struct PagedAttentionTest : public ProblemTest<1> {
    int num_seqs;
    int head_dim = 64;
    int block_size = 16;
    int max_blocks_per_seq = 64; // Max ctx len = 1024
    int total_physical_blocks;
    
    std::vector<float> h_Q, h_K_cache, h_V_cache, h_Out, h_Out_ref;
    std::vector<int> h_block_tables, h_context_lens;

    float *d_Q = nullptr, *d_K_cache = nullptr, *d_V_cache = nullptr, *d_Out = nullptr;
    int *d_block_tables = nullptr, *d_context_lens = nullptr;

    PagedAttentionTest(const TestSize<1>& size) : ProblemTest<1>(size) {
        num_seqs = size.dims[0];
        // Ensure enough physical blocks exist for fragmentation
        total_physical_blocks = num_seqs * max_blocks_per_seq * 2; 
    }

    void generate_test_data(bool check) override {
        h_Q.resize(num_seqs * head_dim);
        h_K_cache.resize(total_physical_blocks * block_size * head_dim);
        h_V_cache.resize(total_physical_blocks * block_size * head_dim);
        h_block_tables.resize(num_seqs * max_blocks_per_seq, -1);
        h_context_lens.resize(num_seqs);
        h_Out.resize(num_seqs * head_dim, 0.0f);
        h_Out_ref.resize(num_seqs * head_dim, 0.0f);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> len_dist(10, max_blocks_per_seq * block_size);
        std::uniform_int_distribution<int> block_dist(0, total_physical_blocks - 1);
        
        for (float& val : h_Q) val = dist(gen);
        for (float& val : h_K_cache) val = dist(gen);
        for (float& val : h_V_cache) val = dist(gen);
        
        for (int i = 0; i < num_seqs; ++i) {
            int ctx_len = len_dist(gen);
            h_context_lens[i] = ctx_len;
            
            int logical_blocks = (ctx_len + block_size - 1) / block_size;
            for (int j = 0; j < logical_blocks; ++j) {
                h_block_tables[i * max_blocks_per_seq + j] = block_dist(gen);
            }
        }

        if (check) {
            cpu_paged_attention(h_Q, h_K_cache, h_V_cache, h_block_tables, h_context_lens, h_Out_ref,
                                num_seqs, head_dim, block_size, max_blocks_per_seq);
        }
    }

    void setup_reference() override {
        cudaMalloc(&d_Q, num_seqs * head_dim * sizeof(float));
        cudaMalloc(&d_K_cache, total_physical_blocks * block_size * head_dim * sizeof(float));
        cudaMalloc(&d_V_cache, total_physical_blocks * block_size * head_dim * sizeof(float));
        cudaMalloc(&d_block_tables, num_seqs * max_blocks_per_seq * sizeof(int));
        cudaMalloc(&d_context_lens, num_seqs * sizeof(int));
        cudaMalloc(&d_Out, num_seqs * head_dim * sizeof(float));
        
        cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K_cache, h_K_cache.data(), h_K_cache.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_cache, h_V_cache.data(), h_V_cache.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_context_lens, h_context_lens.data(), h_context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_reference() override {
        return launch_paged_attention_reference(
            d_Q, d_K_cache, d_V_cache, d_block_tables, d_context_lens, d_Out,
            num_seqs, head_dim, block_size, max_blocks_per_seq);
    }

    void teardown_reference() override {
        cudaFree(d_Q); cudaFree(d_K_cache); cudaFree(d_V_cache);
        cudaFree(d_block_tables); cudaFree(d_context_lens); cudaFree(d_Out);
    }

    void setup_student() override {
        cudaMalloc(&d_Q, num_seqs * head_dim * sizeof(float));
        cudaMalloc(&d_K_cache, total_physical_blocks * block_size * head_dim * sizeof(float));
        cudaMalloc(&d_V_cache, total_physical_blocks * block_size * head_dim * sizeof(float));
        cudaMalloc(&d_block_tables, num_seqs * max_blocks_per_seq * sizeof(int));
        cudaMalloc(&d_context_lens, num_seqs * sizeof(int));
        cudaMalloc(&d_Out, num_seqs * head_dim * sizeof(float));
        
        cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K_cache, h_K_cache.data(), h_K_cache.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_cache, h_V_cache.data(), h_V_cache.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_context_lens, h_context_lens.data(), h_context_lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    std::vector<LaunchConfig> launch_student() override {
        return launch_paged_attention(
            d_Q, d_K_cache, d_V_cache, d_block_tables, d_context_lens, d_Out,
            num_seqs, head_dim, block_size, max_blocks_per_seq);
    }

    void teardown_student() override {
        cudaFree(d_Q); cudaFree(d_K_cache); cudaFree(d_V_cache);
        cudaFree(d_block_tables); cudaFree(d_context_lens); cudaFree(d_Out);
    }

    CorrectnessResult verify() override {
        cudaMemcpy(h_Out.data(), d_Out, num_seqs * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < num_seqs * head_dim; ++i) {
            float diff = std::abs(h_Out_ref[i] - h_Out[i]);
            float magnitude = std::abs(h_Out_ref[i]) + 1e-5f;
            if (diff > 1e-2f && (diff / magnitude) > 1e-2f) {
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
        for (int i = 0; i < num_seqs; ++i) {
            bytes += head_dim * sizeof(float); // Read Q
            int ctx_len = h_context_lens[i];
            bytes += ctx_len * head_dim * sizeof(float) * 2; // Read K, V
            bytes += head_dim * sizeof(float); // Write Out
        }
        return bytes;
    }

    void teardown() override {}
};

Tracer global_tracer(true);

int main(int argc, char** argv) {
    Config<1> config = parse_args<1>(argc, argv);

    // Number of sequences
    std::vector<TestSize<1>> correctness_sizes = {
        {{1}}, {{16}}, {{128}}
    };
    
    std::vector<TestSize<1>> perf_sizes = {
        {{1024}}, {{4096}}
    };

    run_test_suite<1, PagedAttentionTest>("Exercise 31: PagedAttention (KV Cache Management)", config, correctness_sizes, perf_sizes);
    return 0;
}
