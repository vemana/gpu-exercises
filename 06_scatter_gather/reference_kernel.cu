

#include "reference_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_gather(const float* source, float* dest, const int* indices, long long size) {
    global_tracer.trace("Entering launch_reference_gather");
    
    thrust::device_ptr<const float> dev_source(source);
    thrust::device_ptr<const int> dev_indices(indices);
    thrust::device_ptr<float> dev_dest(dest);
    
    global_tracer.trace("Launching thrust::gather");
    thrust::gather(dev_indices, dev_indices + size, dev_source, dev_dest);
    
    global_tracer.trace("Exiting launch_reference_gather");
    
    return {{"thrust::gather", nullptr, 1, 1, 0}};
}
