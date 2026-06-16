#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void map_kernel(const float* a, const float* b, float* c, int size) {
    // TODO: Implement vector addition
}

LaunchMetrics launch_map(const float* a, const float* b, float* c, int size) {
    global_tracer.trace("Entering launch_map (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)map_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching map_kernel (Student)");
    // map_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    
    global_tracer.trace("Exiting launch_map (Student)");
    return {blocksPerGrid, occ};
}
