

#include "kernel.h"

#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void max_pool2d_kernel(const float* input, float* output, long long channels, long long height, long long width) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_max_pool2d(const float* input, float* output, long long channels, long long height, long long width) {
    global_tracer.trace("Entering launch_max_pool2d");
    
    long long out_height = height / 2;
    long long out_width = width / 2;

    dim3 threadsPerBlock(16, 16, 1);
    // TODO: Define grid and block dimensions
    dim3 blocksPerGrid(
        (out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z
    );
    
    global_tracer.trace("Launching max_pool2d_kernel");
    max_pool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, channels, height, width);
    
    global_tracer.trace("Exiting launch_max_pool2d");
    
    return {{"max_pool2d_kernel", (const void*)max_pool2d_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
