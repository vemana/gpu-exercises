#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "../utils/utils.h"
#include "../utils/tracer.h"

LaunchMetrics launch_reference_scan(const float* a, float* c, int size) {
    global_tracer.trace("Entering launch_reference_scan");
    
    thrust::device_ptr<const float> dev_a(a);
    thrust::device_ptr<float> dev_c(c);
    
    global_tracer.trace("Launching thrust::exclusive_scan");
    thrust::exclusive_scan(dev_a, dev_a + size, dev_c);
    
    global_tracer.trace("Exiting launch_reference_scan");
    
    OccupancyMetrics occ;
    occ.is_dummy = true;
    return {1, occ};
}
