#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void histogram_kernel(const int* a, int* bins, int size, int num_bins) {
    // TODO: Implement histogram using atomics
}

LaunchMetrics launch_histogram(const int* a, int* bins, int size, int num_bins) {
    global_tracer.trace("Entering launch_histogram (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)histogram_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching histogram_kernel (Student)");
    // histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, bins, size, num_bins);
    
    global_tracer.trace("Exiting launch_histogram (Student)");
    return {blocksPerGrid, occ};
}
