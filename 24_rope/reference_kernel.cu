#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void rope_reference_kernel(const float* input, float* output, long long batch, long long seq_len, long long hidden_dim) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total_pairs = batch * seq_len * (hidden_dim / 2);
    
    if (idx < total_pairs) {
        long long pairs_per_seq = hidden_dim / 2;
        long long pairs_per_batch = seq_len * pairs_per_seq;
        
        long long b = idx / pairs_per_batch;
        long long s = (idx % pairs_per_batch) / pairs_per_seq;
        long long h = idx % pairs_per_seq;
        
        float freq = powf(10000.0f, -2.0f * (float)h / (float)hidden_dim);
        float theta = (float)s * freq;
        float cos_val = cosf(theta);
        float sin_val = sinf(theta);
        
        long long idx1 = b * seq_len * hidden_dim + s * hidden_dim + 2 * h;
        long long idx2 = b * seq_len * hidden_dim + s * hidden_dim + 2 * h + 1;
        
        float x1 = input[idx1];
        float x2 = input[idx2];
        
        output[idx1] = x1 * cos_val - x2 * sin_val;
        output[idx2] = x1 * sin_val + x2 * cos_val;
    }
}

std::vector<LaunchConfig> launch_reference_rope(const float* input, float* output, long long batch, long long seq_len, long long hidden_dim) {
    global_tracer.trace("Entering launch_reference_rope");
    
    int threadsPerBlock = 256;
    long long total_pairs = batch * seq_len * (hidden_dim / 2);
    long long blocksPerGrid = (total_pairs + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching rope_reference_kernel");
    rope_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, batch, seq_len, hidden_dim);
    
    global_tracer.trace("Exiting launch_reference_rope");
    
    return {{"rope_reference_kernel", (const void*)rope_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
