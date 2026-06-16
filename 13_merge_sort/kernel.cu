#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void merge_sort_kernel(const int* a, int* c, int size) {
    // TODO: Implement merge sort
}

LaunchMetrics launch_merge_sort(const int* a, int* c, int size) {
    global_tracer.trace("Entering launch_merge_sort (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)merge_sort_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching merge_sort_kernel (Student)");
    // merge_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size);
    
    global_tracer.trace("Exiting launch_merge_sort (Student)");
    return {blocksPerGrid, occ};
}
