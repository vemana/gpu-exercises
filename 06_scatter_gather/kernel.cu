

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void gather_kernel(const float* source, float* dest, const int* indices, long long size) {
    // TODO: Implement gather operation
}

std::vector<LaunchConfig> launch_gather(const float* source, float* dest, const int* indices, long long size) {
    global_tracer.trace("Entering launch_gather (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching gather_kernel (Student)");
    // gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(source, dest, indices, size);
    
    global_tracer.trace("Exiting launch_gather (Student)");
    return {{"gather_kernel", (const void*)gather_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
