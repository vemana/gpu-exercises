
#include "kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void sliding_window_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int S, int D, int W) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_sliding_window_attention(
    const float* Q, const float* K, const float* V, float* O,
    int S, int D, int W) {
    
    global_tracer.trace("Entering launch_sliding_window_attention");
    
    // TODO: Define grid and block dimensions
    dim3 block(32);
    dim3 grid(S);
    
    global_tracer.trace("Launching sliding_window_attention_kernel");
    sliding_window_attention_kernel<<<grid, block>>>(Q, K, V, O, S, D, W);
    
    global_tracer.trace("Exiting launch_sliding_window_attention");
    
    return {LaunchConfig{"sliding_window_attention_kernel", (const void*)sliding_window_attention_kernel, (long long)(grid.x), (int)(block.x), (size_t)(0)}};
}
