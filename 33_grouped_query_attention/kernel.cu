
// Note: We use the online softmax trick (similar to FlashAttention) to compute
// attention in O(1) memory, allowing us to keep intermediate accumulators in registers.

#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void gqa_kernel(
    const float* Q, const float* K, const float* V, float* Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_gqa(
    const float* d_Q, const float* d_K, const float* d_V, float* d_Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    
    // TODO: Define grid and block dimensions
    dim3 block(128);
    dim3 grid((seq_len + block.x - 1) / block.x, num_q_heads);
    
    gqa_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_Out, seq_len, num_q_heads, num_kv_heads, head_dim);

    return {{"gqa_kernel", (const void*)gqa_kernel, (long long)grid.x * grid.y, (int)block.x, 0}};
}
