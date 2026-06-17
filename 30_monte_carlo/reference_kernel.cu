

#include "reference_kernel.h"

#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <vector>

#include "../utils/framework.h"

__device__ uint32_t ref_pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float ref_rand_uniform(uint32_t& state) {
    state = ref_pcg_hash(state);
    return (float)(state + 1) / 4294967297.0f;
}

__global__ void ref_monte_carlo_kernel(int N, float S0, float K, float r, float sigma, float T, float* d_sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float payoff = 0.0f;
    if (tid < N) {
        uint32_t state = tid + 12345;
        
        float u1 = ref_rand_uniform(state);
        float u2 = ref_rand_uniform(state);
        
        // Box-Muller transform
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * CUDART_PI_F * u2);
        
        float drift = (r - 0.5f * sigma * sigma) * T;
        float vol = sigma * sqrtf(T);
        float ST = S0 * expf(drift + vol * z);
        
        payoff = fmaxf(ST - K, 0.0f);
    }
    
    // Block-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        payoff += __shfl_down_sync(0xffffffff, payoff, offset);
    }
    
    __shared__ float shared_sum[32];
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (laneId == 0) shared_sum[warpId] = payoff;
    __syncthreads();
    
    if (warpId == 0) {
        float sum = (laneId < (blockDim.x / 32)) ? shared_sum[laneId] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (laneId == 0) {
            atomicAdd(d_sum, sum);
        }
    }
}

std::vector<LaunchConfig> launch_monte_carlo_reference(
    float* d_sum, int N, float S0, float K, float r, float sigma, float T) {
    
    cudaMemset(d_sum, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    ref_monte_carlo_kernel<<<grid, block>>>(N, S0, K, r, sigma, T, d_sum);

    return {{"ref_monte_carlo_kernel", (const void*)ref_monte_carlo_kernel, (long long)grid.x, (int)block.x, 0}};
}
