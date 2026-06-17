

#include "kernel.h"

#include <vector>

#include "../utils/framework.h"

__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_ind, const float* values, const float* x, float* y) {
    // TODO: Implement your kernel here
}

__global__ void dot_kernel(const float* a, const float* b, float* result, int N) {
    // TODO: Implement your kernel here
}

__global__ void init_r_p_kernel(const float* b, const float* Ax, float* r, float* p, int N) {
    // TODO: Implement your kernel here
}

__global__ void update_x_r_kernel(float* x, float* r, const float* p, const float* Ap, const float* r_dot_r, const float* p_Ap, int N) {
    // TODO: Implement your kernel here
}

__global__ void update_p_kernel(float* p, const float* r, const float* r_dot_r_new, const float* r_dot_r_old, int N) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_cg(
    const int* d_row_ptr, const int* d_col_ind, const float* d_values, 
    const float* d_b, float* d_x, int N, int nnz, int num_iters) {
    
    float *d_r, *d_p, *d_Ap;
    cudaMalloc(&d_r, N * sizeof(float));
    cudaMalloc(&d_p, N * sizeof(float));
    cudaMalloc(&d_Ap, N * sizeof(float));
    
    float *d_r_dot_r, *d_r_dot_r_new, *d_p_Ap;
    cudaMalloc(&d_r_dot_r, sizeof(float));
    cudaMalloc(&d_r_dot_r_new, sizeof(float));
    cudaMalloc(&d_p_Ap, sizeof(float));

    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Initial setup: r = b - A*x, p = r
    spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_ind, d_values, d_x, d_Ap);
    init_r_p_kernel<<<grid, block>>>(d_b, d_Ap, d_r, d_p, N);
    
    cudaMemset(d_r_dot_r, 0, sizeof(float));
    dot_kernel<<<grid, block>>>(d_r, d_r, d_r_dot_r, N);

    for (int i = 0; i < num_iters; ++i) {
        // Ap = A * p
        spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_ind, d_values, d_p, d_Ap);
        
        // p_Ap = p dot Ap
        cudaMemset(d_p_Ap, 0, sizeof(float));
        dot_kernel<<<grid, block>>>(d_p, d_Ap, d_p_Ap, N);
        
        // x = x + alpha * p; r = r - alpha * Ap
        update_x_r_kernel<<<grid, block>>>(d_x, d_r, d_p, d_Ap, d_r_dot_r, d_p_Ap, N);
        
        // r_dot_r_new = r dot r
        cudaMemset(d_r_dot_r_new, 0, sizeof(float));
        dot_kernel<<<grid, block>>>(d_r, d_r, d_r_dot_r_new, N);
        
        // p = r + beta * p
        update_p_kernel<<<grid, block>>>(d_p, d_r, d_r_dot_r_new, d_r_dot_r, N);
        
        // Swap r_dot_r and r_dot_r_new pointers for next iter
        float* temp = d_r_dot_r;
        d_r_dot_r = d_r_dot_r_new;
        d_r_dot_r_new = temp;
    }

    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_r_dot_r); cudaFree(d_r_dot_r_new); cudaFree(d_p_Ap);

    // We return one of the configurations to track the core bottleneck (SpMV)
    return {{"spmv_csr_kernel", (const void*)spmv_csr_kernel, (long long)grid.x, (int)block.x, 0}};
}
