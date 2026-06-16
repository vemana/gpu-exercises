#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void spmv_kernel(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, int num_rows) {
    // TODO: Implement sparse matrix-vector multiplication
}

LaunchMetrics launch_spmv(const float* values, const int* col_indices, const int* row_offsets, const float* x, float* y, int num_rows) {
    global_tracer.trace("Entering launch_spmv (Student)");
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    
    OccupancyMetrics occ = calculate_occupancy((const void*)spmv_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Launching spmv_kernel (Student)");
    // spmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(values, col_indices, row_offsets, x, y, num_rows);
    
    global_tracer.trace("Exiting launch_spmv (Student)");
    return {blocksPerGrid, occ};
}
