#include "kernel.h"
#include <math_constants.h>

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

// Reverses the bits of `val`, assuming `val` is a `bits`-bit integer.
__device__ unsigned int reverse_bits(unsigned int val, int bits) {
    unsigned int res = __brev(val); // Reverse all 32 bits
    return res >> (32 - bits);      // Shift down to get only the relevant bits
}

__global__ void bit_reverse_permutation_kernel(cuFloatComplex* x, int N, int log2N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int reversed_tid = reverse_bits(tid, log2N);
        if (tid < reversed_tid) {
            cuFloatComplex temp = x[tid];
            x[tid] = x[reversed_tid];
            x[reversed_tid] = temp;
        }
    }
}

__global__ void fft_stage_kernel(cuFloatComplex* x, int N, int stage) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = 1 << stage;         // Size of the sub-problem in this stage (2, 4, 8...)
    int m2 = m >> 1;            // Half size (1, 2, 4...)
    
    // Each thread computes one butterfly operation. There are N/2 butterfly operations total.
    if (tid < N / 2) {
        // Compute the indices for the butterfly
        int k = tid & (m2 - 1);       // Index within the block
        int j = ((tid - k) << 1) + k; // Global index of the even element
        
        // Compute the twiddle factor: W_m^k = exp(-i * 2 * pi * k / m)
        float theta = -2.0f * CUDART_PI_F * k / m;
        cuFloatComplex W_mk = make_cuFloatComplex(cosf(theta), sinf(theta));
        
        cuFloatComplex t = x[j + m2];
        cuFloatComplex u = x[j];
        
        // Complex multiply: W_mk * t
        cuFloatComplex W_t = make_cuFloatComplex(
            W_mk.x * t.x - W_mk.y * t.y,
            W_mk.x * t.y + W_mk.y * t.x
        );
        
        x[j] = make_cuFloatComplex(u.x + W_t.x, u.y + W_t.y);
        x[j + m2] = make_cuFloatComplex(u.x - W_t.x, u.y - W_t.y);
    }
}

std::vector<LaunchConfig> launch_fft(cuFloatComplex* d_x, int N) {
    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

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
