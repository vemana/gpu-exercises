#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../utils/utils.h"
#include "../utils/tracer.h"
#include <vector>

std::vector<LaunchConfig> launch_reference_reduce(const float* a, float* c, int size) {
    global_tracer.trace("Entering launch_reference_reduce");
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    global_tracer.trace("Allocating CUB temp storage");
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a, c, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    global_tracer.trace("Launching cub::DeviceReduce::Sum");
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a, c, size);
    
    cudaFree(d_temp_storage);
    global_tracer.trace("Exiting launch_reference_reduce");
    
    return {{"cub::DeviceReduce::Sum", nullptr, 1, 1, 0}};
}
