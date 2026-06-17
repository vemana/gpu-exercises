

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void scan_kernel(const float* a, float* c, long long size) {
    // TODO: Implement exclusive prefix sum
}

std::vector<LaunchConfig> launch_scan(const float* a, float* c, long long size) {
    global_tracer.trace("Entering launch_scan (Student)");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching scan_kernel (Student)");
    // scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size);
    
    global_tracer.trace("Exiting launch_scan (Student)");
    return {{"scan_kernel", (const void*)scan_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
