

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void compaction_kernel(const int* a, int* c, int* count, long long size) {
    // TODO: Implement stream compaction
}

std::vector<LaunchConfig> launch_compaction(const int* a, int* c, int* count, long long size) {
    global_tracer.trace("Entering launch_compaction (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching compaction_kernel (Student)");
    // compaction_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, count, size);
    
    global_tracer.trace("Exiting launch_compaction (Student)");
    return {{"compaction_kernel", (const void*)compaction_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
