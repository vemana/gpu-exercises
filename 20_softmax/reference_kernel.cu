#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void softmax_reference_kernel(const float* input, float* output, long long batch_seq, long long vocab_size) {
    long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (b < batch_seq) {
        float max_val = -1e20f;
        for (long long v = 0; v < vocab_size; ++v) {
            float val = input[b * vocab_size + v];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        float sum_exp = 0.0f;
        for (long long v = 0; v < vocab_size; ++v) {
            sum_exp += expf(input[b * vocab_size + v] - max_val);
        }
        
        float inv_sum = 1.0f / sum_exp;
        for (long long v = 0; v < vocab_size; ++v) {
            output[b * vocab_size + v] = expf(input[b * vocab_size + v] - max_val) * inv_sum;
        }
    }
}

std::vector<LaunchConfig> launch_reference_softmax(const float* input, float* output, long long batch_seq, long long vocab_size) {
    global_tracer.trace("Entering launch_reference_softmax");
    
    int threadsPerBlock = 256;
    long long blocksPerGrid = (batch_seq + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching softmax_reference_kernel");
    softmax_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, batch_seq, vocab_size);
    
    global_tracer.trace("Exiting launch_reference_softmax");
    
    return {{"softmax_reference_kernel", (const void*)softmax_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
