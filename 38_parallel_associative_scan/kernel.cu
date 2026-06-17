
#include "kernel.h"

#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void parallel_associative_scan_kernel(
    const float* A, const float* B, const float* x, float* h, int N) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_parallel_associative_scan(
    const float* A, const float* B, const float* x, float* h, int N) {
    
    global_tracer.trace("Entering launch_parallel_associative_scan");
    
    // TODO: Define grid and block dimensions
    dim3 block(N);
    dim3 grid(1);
    
    global_tracer.trace("Launching parallel_associative_scan_kernel");
    parallel_associative_scan_kernel<<<grid, block>>>(A, B, x, h, N);
    
    global_tracer.trace("Exiting launch_parallel_associative_scan");
    
    return {LaunchConfig{"parallel_associative_scan_kernel", (const void*)parallel_associative_scan_kernel, (long long)(grid.x), (int)(block.x), (size_t)(0)}};
}
