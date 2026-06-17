#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

// Reference kernel: 1 thread per row of output (per token in a head in a batch)
// Not optimized, just for reference correctness
__global__ void flash_attention_reference_kernel(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total_rows = B * N * S;
    
    if (idx < total_rows) {
        long long b = idx / (N * S);
        long long n = (idx % (N * S)) / S;
        long long s_q = idx % S;
        
        long long head_offset = (b * N + n) * S * D;
        
        // Find max_val for softmax
        float max_val = -1e20f;
        for (long long s_k = 0; s_k < S; ++s_k) {
            float score = 0.0f;
            for (long long d = 0; d < D; ++d) {
                float q_val = q[head_offset + s_q * D + d];
                float k_val = k[head_offset + s_k * D + d];
                score += q_val * k_val;
            }
            score *= rsqrtf((float)D);
            if (score > max_val) max_val = score;
        }
        
        // Compute sum_exp and exp values
        float sum_exp = 0.0f;
        for (long long s_k = 0; s_k < S; ++s_k) {
            float score = 0.0f;
            for (long long d = 0; d < D; ++d) {
                float q_val = q[head_offset + s_q * D + d];
                float k_val = k[head_offset + s_k * D + d];
                score += q_val * k_val;
            }
            score *= rsqrtf((float)D);
            sum_exp += expf(score - max_val);
        }
        
        // Compute output
        float inv_sum = 1.0f / sum_exp;
        for (long long d = 0; d < D; ++d) {
            float out_val = 0.0f;
            for (long long s_k = 0; s_k < S; ++s_k) {
                float score = 0.0f;
                for (long long i = 0; i < D; ++i) {
                    float q_val = q[head_offset + s_q * D + i];
                    float k_val = k[head_offset + s_k * D + i];
                    score += q_val * k_val;
                }
                score *= rsqrtf((float)D);
                float prob = expf(score - max_val) * inv_sum;
                float v_val = v[head_offset + s_k * D + d];
                out_val += prob * v_val;
            }
            o[head_offset + s_q * D + d] = out_val;
        }
    }
}

std::vector<LaunchConfig> launch_reference_flash_attention(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D) {
    global_tracer.trace("Entering launch_reference_flash_attention");
    
    int threadsPerBlock = 256;
    long long total_rows = B * N * S;
    long long blocksPerGrid = (total_rows + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching flash_attention_reference_kernel");
    flash_attention_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(q, k, v, o, B, N, S, D);
    
    global_tracer.trace("Exiting launch_reference_flash_attention");
    
    return {{"flash_attention_reference_kernel", (const void*)flash_attention_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
