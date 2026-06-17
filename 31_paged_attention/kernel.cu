
// Note: HPC applications typically use double precision or half precision (FP16).
// We use float here for educational consistency.

#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void paged_attention_kernel(
    const float* Q, const float* K_cache, const float* V_cache,
    const int* block_tables, const int* context_lens, float* Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_paged_attention(
    const float* d_Q, const float* d_K_cache, const float* d_V_cache,
    const int* d_block_tables, const int* d_context_lens, float* d_Out,
    int num_seqs, int head_dim, int block_size, int max_blocks_per_seq) {
    
    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid(num_seqs);
    
    int max_ctx_len = 1024;
    int shared_mem_size = head_dim * sizeof(float) + max_ctx_len * sizeof(float);
    
    paged_attention_kernel<<<grid, block, shared_mem_size>>>(
        d_Q, d_K_cache, d_V_cache, d_block_tables, d_context_lens, d_Out,
        num_seqs, head_dim, block_size, max_blocks_per_seq
    );

    return {{"paged_attention_kernel", (const void*)paged_attention_kernel, (long long)grid.x, (int)block.x, (size_t)shared_mem_size}};
}
