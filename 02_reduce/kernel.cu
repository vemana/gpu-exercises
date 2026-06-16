#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void reduce_kernel(const float* a, float* c, int size) {
    // TODO: Implement sum reduction using shared memory
}

LaunchMetrics launch_reduce(const float* a, float* c, int size) {
    global_tracer.trace("Entering launch_reduce (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    size_t dynamic_smem = 0; // Update if using dynamic shared memory
    
    // TODO: Initialize output if necessary (e.g., cudaMemset)
    
    OccupancyMetrics occ = calculate_occupancy((const void*)reduce_kernel, threadsPerBlock, dynamic_smem);
    
    global_tracer.trace("Launching reduce_kernel (Student)");
    // reduce_kernel<<<blocksPerGrid, threadsPerBlock, dynamic_smem>>>(a, c, size);
    
    global_tracer.trace("Exiting launch_reduce (Student)");
    return {blocksPerGrid, occ};
}
