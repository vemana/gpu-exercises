#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "../utils/utils.h"
#include "../utils/tracer.h"

LaunchMetrics launch_reference_merge_sort(const int* a, int* c, int size) {
    global_tracer.trace("Entering launch_reference_merge_sort");
    
    thrust::device_ptr<const int> dev_a(a);
    thrust::device_ptr<int> dev_c(c);
    
    global_tracer.trace("Copying to output array");
    thrust::copy(dev_a, dev_a + size, dev_c);
    
    global_tracer.trace("Launching thrust::sort");
    thrust::sort(dev_c, dev_c + size);
    
    global_tracer.trace("Exiting launch_reference_merge_sort");
    
    OccupancyMetrics occ;
    occ.is_dummy = true;
    return {1, occ};
}
