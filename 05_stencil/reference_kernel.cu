#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void reference_stencil_kernel(const float* a, float* c, int size, int radius) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load central elements
    if (i < size) {
        sdata[radius + tid] = a[i];
    } else {
        sdata[radius + tid] = 0.0f;
    }
    
    // Load halos
    if (tid < radius) {
        // left halo
        int left_idx = blockIdx.x * blockDim.x - radius + tid;
        sdata[tid] = (left_idx >= 0) ? a[left_idx] : 0.0f;
        
        // right halo
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + tid;
        sdata[radius + blockDim.x + tid] = (right_idx < size) ? a[right_idx] : 0.0f;
    }
    
    __syncthreads();
    
    if (i < size) {
        float sum = 0.0f;
        for (int j = -radius; j <= radius; ++j) {
            sum += sdata[radius + tid + j];
        }
        c[i] = sum;
    }
}

std::vector<LaunchConfig> launch_reference_stencil(const float* a, float* c, int size, int radius) {
    global_tracer.trace("Entering launch_reference_stencil");
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    size_t smemSize = (threadsPerBlock + 2 * radius) * sizeof(float);
    
    global_tracer.trace("Launching reference_stencil_kernel");
    reference_stencil_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(a, c, size, radius);
    
    global_tracer.trace("Exiting launch_reference_stencil");
    return {{"reference_stencil_kernel", (const void*)reference_stencil_kernel, blocksPerGrid, threadsPerBlock, smemSize}};
}
