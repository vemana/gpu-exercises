#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void compaction_kernel(const int* a, int* c, int* count, int size) {
    // TODO: Implement stream compaction
}

LaunchMetrics launch_compaction(const int* a, int* c, int* count, int size) {
    global_tracer.trace("Entering launch_compaction (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)compaction_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching compaction_kernel (Student)");
    // compaction_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, count, size);
    
    global_tracer.trace("Exiting launch_compaction (Student)");
    return {blocksPerGrid, occ};
}
