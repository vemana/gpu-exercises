
// A naive row-wise implementation for reference (1 thread per row for simplicity, not performant but correct)

#include "reference_kernel.h"

#include <cmath>
#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void layer_norm_reference_kernel(const float* input, float* output, long long batch_seq, long long hidden_dim) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (b < batch_seq) {
        float sum = 0.0f;
        for (long long h = 0; h < hidden_dim; ++h) {
            sum += input[b * hidden_dim + h];
        }
        float mean = sum / hidden_dim;
        
        float var_sum = 0.0f;
        for (long long h = 0; h < hidden_dim; ++h) {
            float diff = input[b * hidden_dim + h] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden_dim;
        float inv_std = rsqrtf(var + 1e-5f);
        
        for (long long h = 0; h < hidden_dim; ++h) {
            output[b * hidden_dim + h] = (input[b * hidden_dim + h] - mean) * inv_std;
        }
    }
}

std::vector<LaunchConfig> launch_reference_layer_norm(const float* input, float* output, long long batch_seq, long long hidden_dim) {
    global_tracer.trace("Entering launch_reference_layer_norm");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (batch_seq + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching layer_norm_reference_kernel");
    layer_norm_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, batch_seq, hidden_dim);
    
    global_tracer.trace("Exiting launch_reference_layer_norm");
    
    return {{"layer_norm_reference_kernel", (const void*)layer_norm_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
