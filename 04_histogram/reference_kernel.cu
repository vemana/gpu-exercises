#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../utils/utils.h"
#include "../utils/tracer.h"

LaunchMetrics launch_reference_histogram(const int* a, int* bins, int size, int num_bins) {
    global_tracer.trace("Entering launch_reference_histogram");
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    global_tracer.trace("Allocating CUB temp storage");
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, 
        a, bins, num_bins + 1, 0, num_bins, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    global_tracer.trace("Launching cub::DeviceHistogram::HistogramEven");
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, 
        a, bins, num_bins + 1, 0, num_bins, size);
    
    cudaFree(d_temp_storage);
    global_tracer.trace("Exiting launch_reference_histogram");
    
    OccupancyMetrics occ;
    occ.is_dummy = true;
    return {1, occ};
}
