#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "../utils/utils.h"
#include "../utils/tracer.h"

LaunchMetrics launch_reference_map(const float* a, const float* b, float* c, int size) {
    global_tracer.trace("Entering launch_reference_map");
    
    thrust::device_ptr<const float> dev_a(a);
    thrust::device_ptr<const float> dev_b(b);
    thrust::device_ptr<float> dev_c(c);
    
    global_tracer.trace("Launching thrust::transform");
    thrust::transform(dev_a, dev_a + size, dev_b, dev_c, thrust::plus<float>());
    
    global_tracer.trace("Exiting launch_reference_map");
    
    OccupancyMetrics occ;
    occ.is_dummy = true;
    return {1, occ};
}
