#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

// A naive channel-wise implementation for reference (1 block per channel, 1 thread does all reductions)
// This is not performant but correct for the reference baseline.
__global__ void batch_norm_reference_kernel(const float* input, float* output, long long N, long long C, long long H, long long W) {
    long long c = blockIdx.x;
    if (c < C && threadIdx.x == 0) {
        long long spatial_size = H * W;
        long long elements_per_channel = N * spatial_size;
        
        float sum = 0.0f;
        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                sum += input[(n * C + c) * spatial_size + s];
            }
        }
        float mean = sum / elements_per_channel;
        
        float var_sum = 0.0f;
        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                float diff = input[(n * C + c) * spatial_size + s] - mean;
                var_sum += diff * diff;
            }
        }
        float var = var_sum / elements_per_channel;
        float inv_std = rsqrtf(var + 1e-5f);
        
        for (long long n = 0; n < N; ++n) {
            for (long long s = 0; s < spatial_size; ++s) {
                long long idx = (n * C + c) * spatial_size + s;
                output[idx] = (input[idx] - mean) * inv_std;
            }
        }
    }
}

std::vector<LaunchConfig> launch_reference_batch_norm(const float* input, float* output, long long N, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_reference_batch_norm");
    
    int threadsPerBlock = 1;
    long long blocksPerGrid = C;
    
    global_tracer.trace("Launching batch_norm_reference_kernel");
    batch_norm_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, C, H, W);
    
    global_tracer.trace("Exiting launch_reference_batch_norm");
    
    return {{"batch_norm_reference_kernel", (const void*)batch_norm_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
