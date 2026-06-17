
// Note: In reality, A (activations) would be FP16. We use FP32 (float) 
// here to maintain consistency with the other exercises.

#include "kernel.h"

#include <cstdint>
#include <vector>

#include "../utils/framework.h"

__global__ void w8a16_linear_kernel(
    const float* X, const int8_t* W_int8, const float* scales, float* Y,
    int M, int N, int K, int group_size) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_w8a16_linear(
    const float* d_X, const int8_t* d_W_int8, const float* d_scales, float* d_Y,
    int M, int N, int K, int group_size) {
    
    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid(N, M);
    
    w8a16_linear_kernel<<<grid, block>>>(d_X, d_W_int8, d_scales, d_Y, M, N, K, group_size);

    return {{"w8a16_linear_kernel", (const void*)w8a16_linear_kernel, (long long)grid.x * grid.y, (int)block.x, 0}};
}
