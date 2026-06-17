

#include "reference_kernel.h"

#include <algorithm>
#include <cmath>
#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void max_pool2d_reference_kernel(const float* input, float* output, long long channels, long long height, long long width) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    long long out_height = height / 2;
    long long out_width = width / 2;

    long long x = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long y = blockIdx.y * (long long)blockDim.y + threadIdx.y;
    long long c = blockIdx.z * (long long)blockDim.z + threadIdx.z;

    if (x < out_width && y < out_height && c < channels) {
        long long in_x = x * 2;
        long long in_y = y * 2;
        long long channel_offset = c * height * width;

        float val0 = input[channel_offset + in_y * width + in_x];
        float val1 = input[channel_offset + in_y * width + (in_x + 1)];
        float val2 = input[channel_offset + (in_y + 1) * width + in_x];
        float val3 = input[channel_offset + (in_y + 1) * width + (in_x + 1)];

        float max_val = fmaxf(fmaxf(val0, val1), fmaxf(val2, val3));
        
        output[(c * out_height + y) * out_width + x] = max_val;
    }
}

std::vector<LaunchConfig> launch_reference_max_pool2d(const float* input, float* output, long long channels, long long height, long long width) {
    global_tracer.trace("Entering launch_reference_max_pool2d");
    
    long long out_height = height / 2;
    long long out_width = width / 2;

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid(
        (out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z
    );
    
    global_tracer.trace("Launching max_pool2d_reference_kernel");
    max_pool2d_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, channels, height, width);
    
    global_tracer.trace("Exiting launch_reference_max_pool2d");
    
    return {{"max_pool2d_reference_kernel", (const void*)max_pool2d_reference_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
