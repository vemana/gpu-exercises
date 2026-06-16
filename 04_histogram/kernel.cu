#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"
#include <vector>

__global__ void histogram_kernel(const int* a, int* bins, long long size, int num_bins) {
    // TODO: Implement histogram using atomics
}

std::vector<LaunchConfig> launch_histogram(const int* a, int* bins, long long size, int num_bins)
{
    global_tracer.trace("Entering launch_histogram (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    global_tracer.trace("Launching histogram_kernel (Student)");
    // histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, bins, size, num_bins);

    global_tracer.trace("Exiting launch_histogram (Student)");
    return {{"histogram_kernel", (const void*)histogram_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
