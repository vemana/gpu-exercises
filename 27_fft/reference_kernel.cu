#include "reference_kernel.h"
#include <math_constants.h>

__device__ unsigned int ref_reverse_bits(unsigned int val, int bits) {
    unsigned int res = __brev(val);
    return res >> (32 - bits);
}

__global__ void ref_bit_reverse_permutation_kernel(cuFloatComplex* x, int N, int log2N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int reversed_tid = ref_reverse_bits(tid, log2N);
        if (tid < reversed_tid) {
            cuFloatComplex temp = x[tid];
            x[tid] = x[reversed_tid];
            x[reversed_tid] = temp;
        }
    }
}

__global__ void ref_fft_stage_kernel(cuFloatComplex* x, int N, int stage) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = 1 << stage;
    int m2 = m >> 1;
    
    if (tid < N / 2) {
        int k = tid & (m2 - 1);
        int j = ((tid - k) << 1) + k;
        
        float theta = -2.0f * CUDART_PI_F * k / m;
        cuFloatComplex W_mk = make_cuFloatComplex(cosf(theta), sinf(theta));
        
        cuFloatComplex t = x[j + m2];
        cuFloatComplex u = x[j];
        
        cuFloatComplex W_t = make_cuFloatComplex(
            W_mk.x * t.x - W_mk.y * t.y,
            W_mk.x * t.y + W_mk.y * t.x
        );
        
        x[j] = make_cuFloatComplex(u.x + W_t.x, u.y + W_t.y);
        x[j + m2] = make_cuFloatComplex(u.x - W_t.x, u.y - W_t.y);
    }
}

std::vector<LaunchConfig> launch_fft_reference(cuFloatComplex* d_x, int N) {
    int log2N = 0;
    while ((1 << log2N) < N) log2N++;

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    ref_bit_reverse_permutation_kernel<<<grid, block>>>(d_x, N, log2N);

    dim3 butterfly_grid((N / 2 + block.x - 1) / block.x);
    for (int stage = 1; stage <= log2N; ++stage) {
        ref_fft_stage_kernel<<<butterfly_grid, block>>>(d_x, N, stage);
    }

    return {{"ref_fft_stage_kernel", (const void*)ref_fft_stage_kernel, (long long)(butterfly_grid.x), (int)(block.x), 0}};
}
