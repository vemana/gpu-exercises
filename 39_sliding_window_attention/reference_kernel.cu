
// REFERENCE IMPLEMENTATION:
// This kernel is provided for correctness and reference.
// It processes the data according to the mathematical definition of the algorithm.
// Pay attention to the thread indexing and boundary checks.

#include "reference_kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

// Reference Kernel: Block-per-row sliding window attention.
// Explicitly limits memory loads to the causal window [i - W + 1, i].
__global__ void ref_sliding_window_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int S, int D, int W) {
    
    // Simple block-per-row design for reference
    int i = blockIdx.x; // Row i
    int tid = threadIdx.x;
    
    if (i >= S) return;
    
    int start_j = (i - W >= 0) ? (i - W + 1) : 0;
    int end_j = i; // Causal up to i
    
    // Pass 1: compute max
    float max_score = -1e20f;
    if (tid == 0) {
        for (int j = start_j; j <= end_j; ++j) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q[i * D + d] * K[j * D + d];
            }
            score *= rsqrtf((float)D);
            if (score > max_score) max_score = score;
        }
    }
    __syncthreads();
    
    // Use shared memory to broadcast max_score
    __shared__ float s_max_score;
    __shared__ float s_sum_exp;
    if (tid == 0) s_max_score = max_score;
    __syncthreads();
    max_score = s_max_score;
    
    // Pass 2: compute sum_exp
    if (tid == 0) {
        float sum_exp = 0.0f;
        for (int j = start_j; j <= end_j; ++j) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += Q[i * D + d] * K[j * D + d];
            }
            score *= rsqrtf((float)D);
            sum_exp += expf(score - max_score);
        }
        s_sum_exp = sum_exp;
    }
    __syncthreads();
    float sum_exp = s_sum_exp;
    
    // Pass 3: compute final output
    for (int d = tid; d < D; d += blockDim.x) {
        float out_val = 0.0f;
        for (int j = start_j; j <= end_j; ++j) {
            float score = 0.0f;
            for (int d_in = 0; d_in < D; ++d_in) {
                score += Q[i * D + d_in] * K[j * D + d_in];
            }
            score *= rsqrtf((float)D);
            float prob = expf(score - max_score) / sum_exp;
            out_val += prob * V[j * D + d];
        }
        O[i * D + d] = out_val;
    }
}

std::vector<LaunchConfig> launch_reference_sliding_window_attention(
    const float* Q, const float* K, const float* V, float* O,
    int S, int D, int W) {
    
    global_tracer.trace("Entering launch_reference_sliding_window_attention");
    
    dim3 block(32);
    dim3 grid(S);
    
    global_tracer.trace("Launching ref_sliding_window_attention_kernel");
    ref_sliding_window_attention_kernel<<<grid, block>>>(Q, K, V, O, S, D, W);
    
    global_tracer.trace("Exiting launch_reference_sliding_window_attention");
    
    return {LaunchConfig{"ref_sliding_window_attention_kernel", (const void*)ref_sliding_window_attention_kernel, (long long)(grid.x), (int)(block.x), (size_t)(0)}};
}
