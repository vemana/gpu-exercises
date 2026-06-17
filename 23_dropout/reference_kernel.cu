
// Simple deterministic hash for PRNG (PCG Hash) as requested by user

#include "reference_kernel.h"

#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

static __device__ unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__global__ void dropout_reference_kernel(const float* input, float* output, long long size, float drop_prob, float scale) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int hash_val = pcg_hash((unsigned int)idx);
        float rand_val = (float)(hash_val % 10000) / 10000.0f;
        
        if (rand_val < drop_prob) {
            output[idx] = 0.0f;
        } else {
            output[idx] = input[idx] * scale;
        }
    }
}

std::vector<LaunchConfig> launch_reference_dropout(const float* input, float* output, long long size, float drop_prob) {
    global_tracer.trace("Entering launch_reference_dropout");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    float scale = 1.0f / (1.0f - drop_prob);
    
    global_tracer.trace("Launching dropout_reference_kernel");
    dropout_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size, drop_prob, scale);
    
    global_tracer.trace("Exiting launch_reference_dropout");
    
    return {{"dropout_reference_kernel", (const void*)dropout_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
