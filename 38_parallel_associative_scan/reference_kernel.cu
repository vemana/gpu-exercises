
// REFERENCE IMPLEMENTATION:
// This kernel is provided for correctness and reference.
// It processes the data according to the mathematical definition of the algorithm.
// Pay attention to the thread indexing and boundary checks.

#include "reference_kernel.h"

#include <vector>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void ref_parallel_associative_scan_kernel(
    const float* A, const float* B, const float* x, float* h, int N) {
    
    // We assume N <= blockDim.x for this simple reference.
    // A production implementation would handle multi-block scan.
    int tid = threadIdx.x;
    
    // Shared memory for the scan state
    extern __shared__ float s_mem[];
    float* s_A = s_mem;
    float* s_C = s_mem + blockDim.x;
    
    if (tid < N) {
        s_A[tid] = A[tid];
        s_C[tid] = B[tid] * x[tid];
    } else {
        s_A[tid] = 1.0f; // Identity for multiplication
        s_C[tid] = 0.0f; // Identity for addition
    }
    __syncthreads();
    
    // Kogge-Stone Parallel Scan
    // Using the custom associative operator for Mamba recurrence
    // (A1, C1) * (A2, C2) = (A1*A2, A2*C1 + C2)
    for (int offset = 1; offset < N; offset *= 2) {
        float temp_A = s_A[tid];
        float temp_C = s_C[tid];
        
        if (tid >= offset) {
            float prev_A = s_A[tid - offset];
            float prev_C = s_C[tid - offset];
            
            // Associative Operator: S_t = S_t o S_{t-offset}
            temp_A = s_A[tid] * prev_A;
            temp_C = s_A[tid] * prev_C + s_C[tid];
        }
        __syncthreads();
        
        s_A[tid] = temp_A;
        s_C[tid] = temp_C;
        __syncthreads();
    }
    
    if (tid < N) {
        h[tid] = s_C[tid];
    }
}

std::vector<LaunchConfig> launch_reference_parallel_associative_scan(
    const float* A, const float* B, const float* x, float* h, int N) {
    
    global_tracer.trace("Entering launch_reference_parallel_associative_scan");
    
    dim3 block(N); // Assumes N <= 1024
    dim3 grid(1);
    int shared_mem_size = 2 * block.x * sizeof(float);
    
    global_tracer.trace("Launching ref_parallel_associative_scan_kernel");
    ref_parallel_associative_scan_kernel<<<grid, block, shared_mem_size>>>(A, B, x, h, N);
    
    global_tracer.trace("Exiting launch_reference_parallel_associative_scan");
    
    return {LaunchConfig{"ref_parallel_associative_scan_kernel", (const void*)ref_parallel_associative_scan_kernel, (long long)(grid.x), (int)(block.x), (size_t)(shared_mem_size)}};
}
