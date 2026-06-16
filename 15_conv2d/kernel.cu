#include "kernel.h"
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void conv2d_kernel(const float* a, const float* filter, float* c, long long width, long long height) {
    // TODO: Implement 2D convolution
}

std::vector<LaunchConfig> launch_conv2d(const float* a, const float* filter, float* c, long long width, long long height) {
    global_tracer.trace("Entering launch_conv2d (Student)");

    // TODO: Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    global_tracer.trace("Launching conv2d_kernel (Student)");
    // conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, filter, c, width, height);

    global_tracer.trace("Exiting launch_conv2d (Student)");
    return {{"conv2d_kernel", (const void*)conv2d_kernel, static_cast<long long>(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), static_cast<int>(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z), 0}};
}
