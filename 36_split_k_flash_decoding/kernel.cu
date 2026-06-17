
#include "kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void split_k_pass1_kernel(
    const float* q, const float* k, const float* v, 
    float* partial_o, float* partial_m, float* partial_l,
    int S, int D, int split_size) {
    // TODO: Implement your kernel here
}

__global__ void split_k_pass2_kernel(
    float* partial_o, float* partial_m, float* partial_l, float* o,
    int num_splits, int D) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_split_k_flash_decoding(
    const float* q, const float* k, const float* v, float* o,
    float* partial_o, float* partial_m, float* partial_l,
    int S, int D, int split_size) {
    
    global_tracer.trace("Entering launch_split_k_flash_decoding");
    
    int num_splits = (S + split_size - 1) / split_size;
    
    // TODO: Define grid and block dimensions
    dim3 block1(32); // Using warp size as baseline
    dim3 grid1(num_splits);
    
    global_tracer.trace("Launching split_k_pass1_kernel");
    split_k_pass1_kernel<<<grid1, block1>>>(q, k, v, partial_o, partial_m, partial_l, S, D, split_size);
    
    // TODO: Define grid and block dimensions
    dim3 block2(32);
    dim3 grid2(1);
    
    global_tracer.trace("Launching split_k_pass2_kernel");
    split_k_pass2_kernel<<<grid2, block2>>>(partial_o, partial_m, partial_l, o, num_splits, D);
    
    global_tracer.trace("Exiting launch_split_k_flash_decoding");
    
    return {
        LaunchConfig{"split_k_pass1_kernel", (const void*)split_k_pass1_kernel, (long long)(grid1.x), (int)(block1.x), (size_t)(0)},
        LaunchConfig{"split_k_pass2_kernel", (const void*)split_k_pass2_kernel, (long long)(grid2.x), (int)(block2.x), (size_t)(0)}
    };
}
