
// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

#include "kernel.h"

#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <vector>

#include "../utils/framework.h"

__device__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float rand_uniform(uint32_t& state) {
    state = pcg_hash(state);
    return (float)(state + 1) / 4294967297.0f; // strictly in (0, 1)
}

__global__ void monte_carlo_kernel(int N, float S0, float K, float r, float sigma, float T, float* d_sum) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_monte_carlo(
    float* d_sum, int N, float S0, float K, float r, float sigma, float T) {
    
    // Zero out the accumulator
    cudaMemset(d_sum, 0, sizeof(float));

    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    monte_carlo_kernel<<<grid, block>>>(N, S0, K, r, sigma, T, d_sum);

    return {{"monte_carlo_kernel", (const void*)monte_carlo_kernel, (long long)grid.x, (int)block.x, 0}};
}
