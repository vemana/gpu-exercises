
// REFERENCE IMPLEMENTATION:
// This kernel is provided for correctness and reference.
// It processes the data according to the mathematical definition of the algorithm.
// Pay attention to the thread indexing and boundary checks.

#include "reference_kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

// Activation function: Sigmoid Linear Unit (SiLU) / Swish
__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Reference Kernel: Fused point-wise operations.
// Loads X and Gate once, computes SiLU in registers, and stores to O.
__global__ void ref_fused_swiglu_kernel(
    const float* X, const float* gate, float* O, int total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float x_val = X[idx];
        float gate_val = gate[idx];
        
        O[idx] = silu(x_val) * gate_val;
    }
}

std::vector<LaunchConfig> launch_reference_fused_swiglu(
    const float* X, const float* gate, float* O, int total_elements) {
    
    global_tracer.trace("Entering launch_reference_fused_swiglu");
    
    int threads = 256;
    dim3 block(threads);
    dim3 grid((total_elements + threads - 1) / threads);
    
    global_tracer.trace("Launching ref_fused_swiglu_kernel");
    ref_fused_swiglu_kernel<<<grid, block>>>(X, gate, O, total_elements);
    
    global_tracer.trace("Exiting launch_reference_fused_swiglu");
    
    return {LaunchConfig{"ref_fused_swiglu_kernel", (const void*)ref_fused_swiglu_kernel, (long long)(grid.x), (int)(block.x), (size_t)(0)}};
}
