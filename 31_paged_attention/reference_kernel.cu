

#include "reference_kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void ref_paged_attention_kernel(
    const float* Q, const float* K_cache, const float* V_cache,
    const int* block_tables, const int* context_lens, float* Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq) {
    
    int seq_idx = blockIdx.x; // One block per sequence
    int tid = threadIdx.x;
    
    if (seq_idx >= num_seqs) return;
    
    int ctx_len = context_lens[seq_idx];
    const int* seq_block_table = &block_tables[seq_idx * max_blocks_per_seq];
    const float* q = &Q[seq_idx * head_dim];
    
    // Allocate shared memory for Q to be broadcasted to all threads
    extern __shared__ float shared_mem[];
    float* sq = shared_mem; // size: head_dim
    
    if (tid < head_dim) {
        sq[tid] = q[tid];
    }
    __syncthreads();
    
    // We'll compute scores for each token. Since ctx_len can be up to ~1024,
    // and blockDim.x is typically 256, threads will loop over tokens.
    // For simplicity in this exercise, we allocate shared memory for logits.
    // Max ctx_len allowed in this exercise is 1024.
    float* logits = &shared_mem[head_dim]; 
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        int logical_block = t / block_size;
        int physical_block = seq_block_table[logical_block];
        int offset_in_block = t % block_size;
        
        const float* k = &K_cache[(physical_block * block_size + offset_in_block) * head_dim];
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += sq[d] * k[d];
        }
        logits[t] = score * scale;
    }
    __syncthreads();
    
    // Find max logit for numerical stability
    float local_max = -1e20f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        local_max = fmaxf(local_max, logits[t]);
    }
    
    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    __shared__ float s_max;
    if (tid == 0) s_max = local_max; // Not totally safe without block sync/reduce, but okay for single warp?
    // Wait, blockDim > 32, so we need block reduction!
    __shared__ float warp_maxs[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_maxs[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < (blockDim.x / 32)) ? warp_maxs[lane] : -1e20f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) s_max = val;
    }
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        float e = expf(logits[t] - s_max);
        logits[t] = e;
        local_sum += e;
    }
    
    // Block reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    
    __shared__ float s_sum;
    if (warp_id == 0) {
        float val = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) s_sum = val;
    }
    __syncthreads();
    
    // Compute weighted sum of V
    float* out = &Out[seq_idx * head_dim];
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum_v = 0.0f;
        for (int t = 0; t < ctx_len; ++t) {
            int logical_block = t / block_size;
            int physical_block = seq_block_table[logical_block];
            int offset_in_block = t % block_size;
            
            const float* v = &V_cache[(physical_block * block_size + offset_in_block) * head_dim];
            float prob = logits[t] / s_sum;
            sum_v += prob * v[d];
        }
        out[d] = sum_v;
    }
}

std::vector<LaunchConfig> launch_paged_attention_reference(
    const float* d_Q, const float* d_K_cache, const float* d_V_cache,
    const int* d_block_tables, const int* d_context_lens, float* d_Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq) {
    
    dim3 block(256);
    dim3 grid(num_seqs);
    
    int max_ctx_len = 1024;
    int shared_mem_size = head_dim * sizeof(float) + max_ctx_len * sizeof(float);
    
    ref_paged_attention_kernel<<<grid, block, shared_mem_size>>>(
        d_Q, d_K_cache, d_V_cache, d_block_tables, d_context_lens, d_Out,
        num_seqs, head_dim, block_size, max_blocks_per_seq
    );

    return {{"ref_paged_attention_kernel", (const void*)ref_paged_attention_kernel, (long long)grid.x, (int)block.x, (size_t)shared_mem_size}};
}
