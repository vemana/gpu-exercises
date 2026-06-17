

#include "kernel.h"

#include <vector>

#include "../utils/tracer.h"
#include "../utils/utils.h"

__global__ void spmv_kernel(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, long long num_rows) {
    // TODO: Implement sparse matrix-vector multiplication
}

std::vector<LaunchConfig> launch_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, long long num_rows) {
    global_tracer.trace("Entering launch_spmv (Student)");

    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    long long blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    global_tracer.trace("Launching spmv_kernel (Student)");
    // spmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(values, col_indices, row_offsets, x, y, num_rows);

    global_tracer.trace("Exiting launch_spmv (Student)");
    return {{"spmv_kernel", (const void*)spmv_kernel, blocksPerGrid, threadsPerBlock, 0}};
}
