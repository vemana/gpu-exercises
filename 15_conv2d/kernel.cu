#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void conv2d_kernel(const float* a, const float* filter, float* c, int width, int height) {
    // TODO: Implement 2D convolution
}

LaunchMetrics launch_conv2d(const float* a, const float* filter, float* c, int width, int height) {
    global_tracer.trace("Entering launch_conv2d (Student)");
    
    // TODO: Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    OccupancyMetrics occ = calculate_occupancy((const void*)conv2d_kernel, threadsPerBlock.x * threadsPerBlock.y, 0);
    
    global_tracer.trace("Launching conv2d_kernel (Student)");
    // conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, filter, c, width, height);
    
    global_tracer.trace("Exiting launch_conv2d (Student)");
    return {blocksPerGrid.x * blocksPerGrid.y, occ};
}
