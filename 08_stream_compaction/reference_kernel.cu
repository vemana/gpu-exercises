#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

struct is_positive {
    __host__ __device__ bool operator()(const int x) {
        return x > 0;
    }
};

std::vector<LaunchConfig> launch_reference_compaction(const int* a, int* c, int* count, long long size) {
    global_tracer.trace("Entering launch_reference_compaction");
    
    thrust::device_ptr<const int> dev_a(a);
    thrust::device_ptr<int> dev_c(c);
    
    global_tracer.trace("Launching thrust::copy_if");
    auto end_ptr = thrust::copy_if(dev_a, dev_a + size, dev_c, is_positive());
    
    int valid_count = end_ptr - dev_c;
    cudaMemcpy(count, &valid_count, sizeof(int), cudaMemcpyHostToDevice);
    
    global_tracer.trace("Exiting launch_reference_compaction");
    
    return {{"thrust::copy_if", nullptr, 1, 1, 0}};
}
