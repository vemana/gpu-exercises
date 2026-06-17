
// REFERENCE IMPLEMENTATION:
// This kernel is provided for correctness and reference.
// It processes the data according to the mathematical definition of the algorithm.
// Pay attention to the thread indexing and boundary checks.

#include "reference_kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

// Pass 1: Compute partial max (m) and partial sum (l) for a chunk of the sequence.
// Each block handles a subset (split_size) of the sequence S.
__global__ void ref_split_k_pass1_kernel(
    const float* q, const float* k, const float* v, 
    float* partial_o, float* partial_m, float* partial_l,
    int S, int D, int split_size) {
    
    int split_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int start_seq = split_idx * split_size;
    int end_seq = start_seq + split_size;
    if (end_seq > S) end_seq = S;
    
    float max_score = -1e20f;
    float sum_exp = 0.0f;
    
    // For simplicity in the reference, thread 0 does the reduction over the sequence chunk.
    // In an optimized kernel, threads would cooperate within the block.
    if (tid == 0) {
        // Find max
        for (int i = start_seq; i < end_seq; ++i) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * k[i * D + d];
            }
            score *= rsqrtf((float)D);
            if (score > max_score) max_score = score;
        }
        
        // Compute sum of exps
        for (int i = start_seq; i < end_seq; ++i) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * k[i * D + d];
            }
            score *= rsqrtf((float)D);
            sum_exp += expf(score - max_score);
        }
        
        partial_m[split_idx] = max_score;
        partial_l[split_idx] = sum_exp;
    }
    __syncthreads();
    
    // Output partial vector (D elements)
    for (int d = tid; d < D; d += blockDim.x) {
        float out_val = 0.0f;
        for (int i = start_seq; i < end_seq; ++i) {
            float score = 0.0f;
            for (int d_in = 0; d_in < D; ++d_in) {
                score += q[d_in] * k[i * D + d_in];
            }
            score *= rsqrtf((float)D);
            float prob = expf(score - max_score);
            out_val += prob * v[i * D + d];
        }
        partial_o[split_idx * D + d] = out_val;
    }
}

// Pass 2: Combine the partial results from Pass 1 into the final attention output.
// Reduces across the split dimension using the global max logic.
__global__ void ref_split_k_pass2_kernel(
    float* partial_o, float* partial_m, float* partial_l, float* o,
    int num_splits, int D) {
    
    int tid = threadIdx.x;
    
    __shared__ float global_max;
    __shared__ float global_sum;
    
    if (tid == 0) {
        float max_val = -1e20f;
        for (int i = 0; i < num_splits; ++i) {
            if (partial_m[i] > max_val) max_val = partial_m[i];
        }
        global_max = max_val;
        
        float sum_val = 0.0f;
        for (int i = 0; i < num_splits; ++i) {
            sum_val += partial_l[i] * expf(partial_m[i] - global_max);
        }
        global_sum = sum_val;
    }
    __syncthreads();
    
    for (int d = tid; d < D; d += blockDim.x) {
        float final_out = 0.0f;
        for (int i = 0; i < num_splits; ++i) {
            float weight = expf(partial_m[i] - global_max);
            final_out += partial_o[i * D + d] * weight;
        }
        o[d] = final_out / global_sum;
    }
}

std::vector<LaunchConfig> launch_reference_split_k_flash_decoding(
    const float* q, const float* k, const float* v, float* o,
    float* partial_o, float* partial_m, float* partial_l,
    int S, int D, int split_size) {
    
    global_tracer.trace("Entering launch_reference_split_k_flash_decoding");
    
    int num_splits = (S + split_size - 1) / split_size;
    dim3 block1(32);
    dim3 grid1(num_splits);
    
    global_tracer.trace("Launching ref_split_k_pass1_kernel");
    ref_split_k_pass1_kernel<<<grid1, block1>>>(q, k, v, partial_o, partial_m, partial_l, S, D, split_size);
    
    dim3 block2(32);
    dim3 grid2(1);
    
    global_tracer.trace("Launching ref_split_k_pass2_kernel");
    ref_split_k_pass2_kernel<<<grid2, block2>>>(partial_o, partial_m, partial_l, o, num_splits, D);
    
    global_tracer.trace("Exiting launch_reference_split_k_flash_decoding");
    
    return {
        LaunchConfig{"ref_split_k_pass1_kernel", (const void*)ref_split_k_pass1_kernel, (long long)(grid1.x), (int)(block1.x), (size_t)(0)},
        LaunchConfig{"ref_split_k_pass2_kernel", (const void*)ref_split_k_pass2_kernel, (long long)(grid2.x), (int)(block2.x), (size_t)(0)}
    };
}
