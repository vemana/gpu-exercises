#include "kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void layer_norm_kernel(const float* input, float* output, long long batch_seq, long long hidden_dim) {
    long long b = blockIdx.x;
    if (b < batch_seq) {
        // student implements layer norm here using block reduction or loop
    }
}

std::vector<LaunchConfig> launch_layer_norm(const float* input, float* output, long long batch_seq, long long hidden_dim) {
    global_tracer.trace("Entering launch_layer_norm");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = batch_seq;
    
    global_tracer.trace("Launching layer_norm_kernel");
    layer_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, batch_seq, hidden_dim);
    
    global_tracer.trace("Exiting launch_layer_norm");
    
    return {{"layer_norm_kernel", (const void*)layer_norm_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
