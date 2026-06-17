

#include "reference_kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void ref_gqa_kernel(
    const float* Q, const float* K, const float* V, float* Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_idx = blockIdx.y;
    
    if (seq_idx >= seq_len) return;
    
    int group_ratio = num_q_heads / num_kv_heads;
    int kv_idx = q_idx / group_ratio;
    
    const float* my_q = &Q[(q_idx * seq_len + seq_idx) * head_dim];
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    float m = -1e20f;
    float l = 0.0f;
    
    // We assume head_dim = 64 for static array allocation in registers
    float out_vec[64] = {0.0f};
    
    for (int j = 0; j < seq_len; ++j) {
        const float* my_k = &K[(kv_idx * seq_len + j) * head_dim];
        const float* my_v = &V[(kv_idx * seq_len + j) * head_dim];
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += my_q[d] * my_k[d];
        }
        score *= scale;
        
        float m_new = fmaxf(m, score);
        float exp_diff = expf(m - m_new);
        float exp_val = expf(score - m_new);
        
        l = l * exp_diff + exp_val;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = out_vec[d] * exp_diff + exp_val * my_v[d];
        }
        m = m_new;
    }
    
    float* my_out = &Out[(q_idx * seq_len + seq_idx) * head_dim];
    for (int d = 0; d < head_dim; ++d) {
        my_out[d] = out_vec[d] / l;
    }
}

std::vector<LaunchConfig> launch_gqa_reference(
    const float* d_Q, const float* d_K, const float* d_V, float* d_Out,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    
    dim3 block(128);
    dim3 grid((seq_len + block.x - 1) / block.x, num_q_heads);
    
    ref_gqa_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_Out, seq_len, num_q_heads, num_kv_heads, head_dim);

    return {{"ref_gqa_kernel", (const void*)ref_gqa_kernel, (long long)grid.x * grid.y, (int)block.x, 0}};
}
