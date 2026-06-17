

#include "kernel.h"

#include <math_functions.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/framework.h"
#include "../utils/tracer.h"

__global__ void softmax_kernel(const float* input, float* output, long long batch_seq, long long vocab_size) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_softmax(const float* input, float* output, long long batch_seq, long long vocab_size) {
    global_tracer.trace("Entering launch_softmax");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = batch_seq;
    
    global_tracer.trace("Launching softmax_kernel");
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, batch_seq, vocab_size);
    
    global_tracer.trace("Exiting launch_softmax");
    
    return {{"softmax_kernel", (const void*)softmax_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
