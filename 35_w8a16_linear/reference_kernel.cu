
#include "reference_kernel.h"

#include <cstdint>
#include <vector>

#include "../utils/framework.h"

__global__ void ref_w8a16_linear_kernel(
    const float* X, const int8_t* W_int8, const float* scales, float* Y,
    int M, int N, int K, int group_size) {
    
    // Grid: (N, M)
    // Each block computes exactly 1 element of the output matrix Y: Y[m, n]
    int n = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    float local_sum = 0.0f;
    
    // K is typically large (e.g. 4096), so 256 threads loop over it
    for (int k = tid; k < K; k += blockDim.x) {
        int group_id = k / group_size;
        
        // 1. Fetch the scale for this group
        float scale = scales[n * (K / group_size) + group_id];
        
        // 2. Fetch the int8 weight
        int8_t w_quant = W_int8[n * K + k];
        
        // 3. Dequantize on-the-fly in the register
        float w_float = (float)w_quant * scale;
        
        // 4. Fetch the float activation and compute MAC
        float x_val = X[m * K + k];
        local_sum += x_val * w_float;
    }
    
    // Block-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    __shared__ float shared_sum[32];
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            Y[m * N + n] = sum;
        }
    }
}

std::vector<LaunchConfig> launch_w8a16_linear_reference(
    const float* d_X, const int8_t* d_W_int8, const float* d_scales, float* d_Y,
    int M, int N, int K, int group_size) {
    
    dim3 block(256);
    dim3 grid(N, M);
    
    ref_w8a16_linear_kernel<<<grid, block>>>(d_X, d_W_int8, d_scales, d_Y, M, N, K, group_size);

    return {{"ref_w8a16_linear_kernel", (const void*)ref_w8a16_linear_kernel, (long long)grid.x * grid.y, (int)block.x, 0}};
}
