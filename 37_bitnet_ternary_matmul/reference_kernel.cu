
// REFERENCE IMPLEMENTATION:
// This kernel is provided for correctness and reference.
// It processes the data according to the mathematical definition of the algorithm.
// Pay attention to the thread indexing and boundary checks.

#include "reference_kernel.h"

#include <cstdint>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

// Reference Kernel: Decodes the 2-bit weights directly in registers to bypass FMA units.
// Computes ternary dot products via conditional accumulations.
__global__ void ref_bitnet_ternary_matmul_kernel(
    const float* A, const uint8_t* W_packed, float* C,
    int M, int N, int K) {
    
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m < M && n < N) {
        float sum = 0.0f;
        
        // K is a multiple of 4, so we process 4 elements per byte
        int k_packed_limit = K / 4;
        
        for (int k_idx = 0; k_idx < k_packed_limit; ++k_idx) {
            uint8_t packed_val = W_packed[k_idx * N + n];
            
            for (int i = 0; i < 4; ++i) {
                int k = k_idx * 4 + i;
                float a_val = A[m * K + k];
                
                uint8_t bits = (packed_val >> (i * 2)) & 0x3;
                
                // 1 -> +1, 2 -> -1, 0 -> 0
                if (bits == 1) {
                    sum += a_val;
                } else if (bits == 2) {
                    sum -= a_val;
                }
            }
        }
        
        C[m * N + n] = sum;
    }
}

std::vector<LaunchConfig> launch_reference_bitnet_ternary_matmul(
    const float* A, const uint8_t* W_packed, float* C,
    int M, int N, int K) {
    
    global_tracer.trace("Entering launch_reference_bitnet_ternary_matmul");
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    global_tracer.trace("Launching ref_bitnet_ternary_matmul_kernel");
    ref_bitnet_ternary_matmul_kernel<<<grid, block>>>(A, W_packed, C, M, N, K);
    
    global_tracer.trace("Exiting launch_reference_bitnet_ternary_matmul");
    
    return {LaunchConfig{"ref_bitnet_ternary_matmul_kernel", (const void*)ref_bitnet_ternary_matmul_kernel, (long long)(grid.x * grid.y), (int)(block.x * block.y), (size_t)(0)}};
}
