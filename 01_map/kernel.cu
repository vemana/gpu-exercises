#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void map_kernel(const float* a, const float* b, float* c, long long size) {
  /*
  long long lane = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = blockDim.x * gridDim.x;
  for (;lane < size; lane += stride) {
    c[lane] = a[lane] + b[lane];
  }
  */
}

std::vector<LaunchConfig> launch_map(const float* a, const float* b, float* c, long long size) {
    global_tracer.trace("Entering launch_map (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching map_kernel (Student)");
    map_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    
    global_tracer.trace("Exiting launch_map (Student)");
    return {{"map_kernel", (const void*)map_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
