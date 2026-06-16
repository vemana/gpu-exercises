#include "kernel.h"
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void merge_sort_kernel(const int* a, int* c, long long size) {
    // TODO: Implement merge sort
}

std::vector<LaunchConfig> launch_merge_sort(const int* a, int* c, long long size) {
    global_tracer.trace("Entering launch_merge_sort (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    global_tracer.trace("Launching merge_sort_kernel (Student)");
    // merge_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size);

    global_tracer.trace("Exiting launch_merge_sort (Student)");
    return {{"merge_sort_kernel", (const void*)merge_sort_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
