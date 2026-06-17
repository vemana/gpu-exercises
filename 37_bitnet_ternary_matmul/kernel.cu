
#include "kernel.h"

#include <cstdint>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void bitnet_ternary_matmul_kernel(
    const float* A, const uint8_t* W_packed, float* C,
    int M, int N, int K) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_bitnet_ternary_matmul(
    const float* A, const uint8_t* W_packed, float* C,
    int M, int N, int K) {
    
    global_tracer.trace("Entering launch_bitnet_ternary_matmul");
    
    // TODO: Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    global_tracer.trace("Launching bitnet_ternary_matmul_kernel");
    bitnet_ternary_matmul_kernel<<<grid, block>>>(A, W_packed, C, M, N, K);
    
    global_tracer.trace("Exiting launch_bitnet_ternary_matmul");
    
    return {LaunchConfig{"bitnet_ternary_matmul_kernel", (const void*)bitnet_ternary_matmul_kernel, (long long)(grid.x * grid.y), (int)(block.x * block.y), (size_t)(0)}};
}
