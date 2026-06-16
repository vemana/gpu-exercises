#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void scan_kernel(const float* a, float* c, int size) {
    // TODO: Implement exclusive prefix sum
}

LaunchMetrics launch_scan(const float* a, float* c, int size) {
    global_tracer.trace("Entering launch_scan (Student)");
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)scan_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching scan_kernel (Student)");
    // scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size);
    
    global_tracer.trace("Exiting launch_scan (Student)");
    return {blocksPerGrid, occ};
}
