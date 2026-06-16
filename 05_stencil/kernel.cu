#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void stencil_kernel(const float* a, float* c, int size, int radius) {
    // TODO: Implement 1D stencil using shared memory
}

LaunchMetrics launch_stencil(const float* a, float* c, int size, int radius) {
    global_tracer.trace("Entering launch_stencil (Student)");
    
    // TODO: Define grid and block dimensions, and dynamic shared memory
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)stencil_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching stencil_kernel (Student)");
    // stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size, radius);
    
    global_tracer.trace("Exiting launch_stencil (Student)");
    return {blocksPerGrid, occ};
}
