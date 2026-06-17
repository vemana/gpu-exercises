

#include "reference_kernel.h"

#include <cub/cub.cuh>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_histogram(const int* a, int* bins, long long size, int num_bins) {
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
    
    return {{"cub::DeviceHistogram::HistogramEven", nullptr, 1, 1, 0}};
}
