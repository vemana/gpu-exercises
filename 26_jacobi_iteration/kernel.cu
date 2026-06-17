
// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

#include "kernel.h"

#include <vector>

#include "../utils/framework.h"

__global__ void jacobi_kernel(const float* u_old, float* u_new, int H, int W) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_jacobi(float* d_u, float* d_u_tmp, int H, int W, int num_iters) {
    // TODO: Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    float* current = d_u;
    float* next = d_u_tmp;

    for (int i = 0; i < num_iters; ++i) {
        jacobi_kernel<<<grid, block>>>(current, next, H, W);
        std::swap(current, next);
    }

    if (num_iters % 2 != 0) {
        cudaMemcpy(d_u, d_u_tmp, H * W * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {{"jacobi_kernel", (const void*)jacobi_kernel, (long long)(grid.x * grid.y * grid.z), (int)(block.x * block.y * block.z), 0}};
}
