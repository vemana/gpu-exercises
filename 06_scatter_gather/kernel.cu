#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void gather_kernel(const float* source, float* dest, const int* indices, int size) {
    // TODO: Implement gather operation
}

LaunchMetrics launch_gather(const float* source, float* dest, const int* indices, int size) {
    global_tracer.trace("Entering launch_gather (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)gather_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching gather_kernel (Student)");
    // gather_kernel<<<blocksPerGrid, threadsPerBlock>>>(source, dest, indices, size);
    
    global_tracer.trace("Exiting launch_gather (Student)");
    return {blocksPerGrid, occ};
}
