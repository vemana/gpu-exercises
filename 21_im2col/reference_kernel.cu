#include "reference_kernel.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "../utils/tracer.h"

__global__ void im2col_reference_kernel(const float* input, float* output, long long C, long long H, long long W) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total_elements = C * H * W;
    if (idx < total_elements) {
        long long c = idx / (H * W);
        long long hw = idx % (H * W);
        long long y = hw / W;
        long long x = hw % W;

        long long col_col = y * W + x;

        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                long long col_row = c * 9 + (fy + 1) * 3 + (fx + 1);
                long long in_y = y + fy;
                long long in_x = x + fx;
                
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    val = input[c * H * W + in_y * W + in_x];
                }
                output[col_row * (H * W) + col_col] = val;
            }
        }
    }
}

std::vector<LaunchConfig> launch_reference_im2col(const float* input, float* output, long long C, long long H, long long W) {
    global_tracer.trace("Entering launch_reference_im2col");
    
    int threadsPerBlock = 256;
    long long total_elements = C * H * W;
    long long blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    global_tracer.trace("Launching im2col_reference_kernel");
    im2col_reference_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, C, H, W);
    
    global_tracer.trace("Exiting launch_reference_im2col");
    
    return {{"im2col_reference_kernel", (const void*)im2col_reference_kernel, (int)blocksPerGrid, threadsPerBlock, 0}};
}
