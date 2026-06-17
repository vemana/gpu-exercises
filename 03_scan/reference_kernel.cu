

#include "reference_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

std::vector<LaunchConfig> launch_reference_scan(const float* a, float* c, long long size) {
    global_tracer.trace("Entering launch_reference_scan");
    
    thrust::device_ptr<const float> dev_a(a);
    thrust::device_ptr<float> dev_c(c);
    
    global_tracer.trace("Launching thrust::exclusive_scan");
    thrust::exclusive_scan(dev_a, dev_a + size, dev_c);
    
    global_tracer.trace("Exiting launch_reference_scan");
    
    return {{"thrust::exclusive_scan", nullptr, 1, 1, 0}};
}
