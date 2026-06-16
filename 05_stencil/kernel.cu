#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"
#include <vector>

__global__ void stencil_kernel(const float* a, float* c, long long size, int radius) {
    // TODO: Implement 1D stencil using shared memory
}

std::vector<LaunchConfig> launch_stencil(const float* a, float* c, long long size, int radius) {
    global_tracer.trace("Entering launch_stencil (Student)");
    
    // TODO: Define grid and block dimensions, and dynamic shared memory
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching stencil_kernel (Student)");
    // stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size, radius);
    
    global_tracer.trace("Exiting launch_stencil (Student)");
    return {{"stencil_kernel", (const void*)stencil_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
