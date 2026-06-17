

#include "kernel.h"

#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void depthwise_conv2d_kernel(const float* input, const float* filter, float* output, long long C, long long H, long long W) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_depthwise_conv2d(const float* input, const float* filter, float* output, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_depthwise_conv2d");
    
    dim3 threadsPerBlock(16, 16, 1);
    // TODO: Define grid and block dimensions
    dim3 blocksPerGrid((W + 15) / 16, (H + 15) / 16, C);
    
    global_tracer.trace("Launching depthwise_conv2d_kernel");
    depthwise_conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, filter, output, C, H, W);
    
    global_tracer.trace("Exiting launch_depthwise_conv2d");
    
    return {{"depthwise_conv2d_kernel", (const void*)depthwise_conv2d_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
