

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void segmented_scan_kernel(const float* a, const int* flags, float* c, long long size) {
    // TODO: Implement segmented scan
}

std::vector<LaunchConfig> launch_segmented_scan(const float* a, const int* flags, float* c, long long size) {
    global_tracer.trace("Entering launch_segmented_scan (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching segmented_scan_kernel (Student)");
    // segmented_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, flags, c, size);
    
    global_tracer.trace("Exiting launch_segmented_scan (Student)");
    return {{"segmented_scan_kernel", (const void*)segmented_scan_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
