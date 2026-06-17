

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void reduce_kernel(const float* a, float* c, long long size) {
    // TODO: Implement sum reduction using shared memory
}

std::vector<LaunchConfig> launch_reduce(const float* a, float* c, long long size) {
    global_tracer.trace("Entering launch_reduce (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    size_t dynamic_smem = 0; // Update if using dynamic shared memory

    // TODO: Initialize output if necessary (e.g., cudaMemset)

    global_tracer.trace("Launching reduce_kernel (Student)");
    // reduce_kernel<<<blocksPerGrid, threadsPerBlock, dynamic_smem>>>(a, c, size);

    global_tracer.trace("Exiting launch_reduce (Student)");
    return {{"reduce_kernel", (const void*)reduce_kernel, blocksPerGrid, threadsPerBlock, dynamic_smem}};
}
