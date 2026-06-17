#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void depthwise_conv2d_reference_kernel(const float* input, const float* filter, float* output, long long C, long long H, long long W) {
    long long x = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long y = blockIdx.y * (long long)blockDim.y + threadIdx.y;
    long long c = blockIdx.z;

    if (x < W && y < H && c < C) {
        float sum = 0.0f;
        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                long long in_y = y + fy;
                long long in_x = x + fx;
                
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    float pixel = input[c * H * W + in_y * W + in_x];
                    float weight = filter[c * 9 + (fy + 1) * 3 + (fx + 1)];
                    sum += pixel * weight;
                }
            }
        }
        output[c * H * W + y * W + x] = sum;
    }
}

std::vector<LaunchConfig> launch_reference_depthwise_conv2d(const float* input, const float* filter, float* output, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_reference_depthwise_conv2d");
    
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((W + 15) / 16, (H + 15) / 16, C);
    
    global_tracer.trace("Launching depthwise_conv2d_reference_kernel");
    depthwise_conv2d_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, filter, output, C, H, W);
    
    global_tracer.trace("Exiting launch_reference_depthwise_conv2d");
    
    return {{"depthwise_conv2d_reference_kernel", (const void*)depthwise_conv2d_reference_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
