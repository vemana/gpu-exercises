
// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

// Reverses the bits of `val`, assuming `val` is a `bits`-bit integer.

#include "kernel.h"

#include <math_constants.h>
#include <vector>

#include "../utils/framework.h"

__device__ unsigned int reverse_bits(unsigned int val, int bits) {
    unsigned int res = __brev(val); // Reverse all 32 bits
    return res >> (32 - bits);      // Shift down to get only the relevant bits
}

__global__ void bit_reverse_permutation_kernel(cuFloatComplex* x, int N, int log2N) {
    // TODO: Implement your kernel here
}

__global__ void fft_stage_kernel(cuFloatComplex* x, int N, int stage) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_fft(cuFloatComplex* d_x, int N) {
    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // 1. Bit-reversal permutation
    bit_reverse_permutation_kernel<<<grid, block>>>(d_x, N, log2N);

    // 2. Cooley-Tukey FFT stages
    dim3 butterfly_grid((N / 2 + block.x - 1) / block.x);
    for (int stage = 1; stage <= log2N; ++stage) {
        fft_stage_kernel<<<butterfly_grid, block>>>(d_x, N, stage);
    }

    // We return the config for the last stage just to track metrics
    return {{"fft_stage_kernel", (const void*)fft_stage_kernel, (long long)(butterfly_grid.x), (int)(block.x), 0}};
}
