

#include "reference_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <vector>

#include <cuda_runtime.h>

#include "../utils/tracer.h"
#include "../utils/utils.h"

struct SegmentedScanOp {
    __host__ __device__
    thrust::tuple<float, int> operator()(const thrust::tuple<float, int>& a, 
                                         const thrust::tuple<float, int>& b) const {
        float val_a = thrust::get<0>(a);
        int flag_a = thrust::get<1>(a);
        
        float val_b = thrust::get<0>(b);
        int flag_b = thrust::get<1>(b);
        
        return thrust::make_tuple(
            flag_b ? val_b : val_a + val_b,
            flag_a | flag_b
        );
    }
};

std::vector<LaunchConfig> launch_reference_segmented_scan(const float* a, const int* flags, float* c, long long size) {
    global_tracer.trace("Entering launch_reference_segmented_scan");
    
    thrust::device_ptr<const float> dev_a(a);
    thrust::device_ptr<const int> dev_flags(flags);
    thrust::device_ptr<float> dev_c(c);
    
    auto in_zip = thrust::make_zip_iterator(thrust::make_tuple(dev_a, dev_flags));
    auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(dev_c, thrust::make_discard_iterator()));
    
    global_tracer.trace("Launching thrust::inclusive_scan");
    thrust::inclusive_scan(in_zip, in_zip + size, out_zip, SegmentedScanOp());
    
    global_tracer.trace("Exiting launch_reference_segmented_scan");
    
    return {{"thrust::inclusive_scan", nullptr, 1, 1, 0}};
}
