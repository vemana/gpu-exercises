

#include "reference_kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void ref_moe_routing_kernel(
    const float* logits, int* top_experts, float* routing_weights, 
    int num_tokens, int num_experts, int K) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;
    
    // Each thread processes 1 token's logits
    const float* my_logits = &logits[tid * num_experts];
    int* my_experts = &top_experts[tid * K];
    float* my_weights = &routing_weights[tid * K];
    
    // Hardcoded for Top-2 out of E experts, standard for Mixtral 8x7B.
    int best_e = -1, second_e = -1;
    float best_val = -1e20f, second_val = -1e20f;
    
    for (int e = 0; e < num_experts; ++e) {
        float val = my_logits[e];
        if (val > best_val) {
            second_val = best_val;
            second_e = best_e;
            best_val = val;
            best_e = e;
        } else if (val > second_val) {
            second_val = val;
            second_e = e;
        }
    }
    
    // Compute Softmax strictly over the top K logits
    float max_val = best_val;
    float exp1 = expf(best_val - max_val);
    float exp2 = expf(second_val - max_val);
    float sum_exp = exp1 + exp2;
    
    my_experts[0] = best_e;
    my_experts[1] = second_e;
    
    my_weights[0] = exp1 / sum_exp;
    my_weights[1] = exp2 / sum_exp;
}

std::vector<LaunchConfig> launch_moe_routing_reference(
    const float* d_logits, int* d_top_experts, float* d_routing_weights, 
    int num_tokens, int num_experts, int K) {
    
    dim3 block(256);
    dim3 grid((num_tokens + block.x - 1) / block.x);
    
    ref_moe_routing_kernel<<<grid, block>>>(d_logits, d_top_experts, d_routing_weights, num_tokens, num_experts, K);

    return {{"ref_moe_routing_kernel", (const void*)ref_moe_routing_kernel, (long long)grid.x, (int)block.x, 0}};
}
