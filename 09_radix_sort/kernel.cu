#include "kernel.h"
#include <vector>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void radix_sort_kernel(const int* a, int* c, int size) {
    // TODO: Implement radix sort
}

std::vector<LaunchConfig> launch_radix_sort(const int* a, int* c, int size) {
    global_tracer.trace("Entering launch_radix_sort (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    global_tracer.trace("Launching radix_sort_kernel (Student)");
    // radix_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, size);

    global_tracer.trace("Exiting launch_radix_sort (Student)");
    return {{"radix_sort_kernel", (const void*)radix_sort_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
