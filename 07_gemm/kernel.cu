#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
    // TODO: Implement tiled matrix multiplication using shared memory
}

std::vector<LaunchConfig> launch_gemm(const float* a, const float* b, float* c, int n) {
    global_tracer.trace("Entering launch_gemm (Student)");
    
    // TODO: Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    global_tracer.trace("Launching gemm_kernel (Student)");
    // gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
    
    global_tracer.trace("Exiting launch_gemm (Student)");
    return {{"gemm_kernel", (const void*)gemm_kernel, (int)(blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z), (int)(threadsPerBlock.x * threadsPerBlock.y), 0}};
}
