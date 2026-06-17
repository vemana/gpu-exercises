#include "kernel.h"

// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

__global__ void jacobi_kernel(const float* u_old, float* u_new, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < W - 1 && y > 0 && y < H - 1) {
        u_new[y * W + x] = 0.25f * (u_old[(y - 1) * W + x] + u_old[(y + 1) * W + x] + 
                                    u_old[y * W + x - 1] + u_old[y * W + x + 1]);
    } else if (x < W && y < H) {
        u_new[y * W + x] = u_old[y * W + x];
    }
}

std::vector<LaunchConfig> launch_jacobi(float* d_u, float* d_u_tmp, int H, int W, int num_iters) {
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
