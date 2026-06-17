#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void flash_attention_kernel(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D) {
    // student implements flash attention (or standard attention) here
}

std::vector<LaunchConfig> launch_flash_attention(const float* q, const float* k, const float* v, float* o, long long B, long long N, long long S, long long D) {
    global_tracer.trace("Entering launch_flash_attention");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = B * N * S;
    
    global_tracer.trace("Launching flash_attention_kernel");
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(q, k, v, o, B, N, S, D);
    
    global_tracer.trace("Exiting launch_flash_attention");
    
    return {{"flash_attention_kernel", (const void*)flash_attention_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
