#include "reference_kernel.h"

__global__ void ref_spmv_csr_kernel(int num_rows, const int* row_ptr, const int* col_ind, const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; ++j) {
            dot += values[j] * x[col_ind[j]];
        }
        y[row] = dot;
    }
}

__global__ void ref_dot_kernel(const float* a, const float* b, float* result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (tid < N) {
        sum = a[tid] * b[tid];
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float shared_sum[32];
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (laneId == 0) shared_sum[warpId] = sum;
    __syncthreads();
    
    if (warpId == 0) {
        sum = (laneId < (blockDim.x / 32)) ? shared_sum[laneId] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (laneId == 0) {
            atomicAdd(result, sum);
        }
    }
}

__global__ void ref_init_r_p_kernel(const float* b, const float* Ax, float* r, float* p, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float val = b[tid] - Ax[tid];
        r[tid] = val;
        p[tid] = val;
    }
}

__global__ void ref_update_x_r_kernel(float* x, float* r, const float* p, const float* Ap, const float* r_dot_r, const float* p_Ap, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float alpha = (*r_dot_r) / (*p_Ap);
        x[tid] += alpha * p[tid];
        r[tid] -= alpha * Ap[tid];
    }
}

__global__ void ref_update_p_kernel(float* p, const float* r, const float* r_dot_r_new, const float* r_dot_r_old, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float beta = (*r_dot_r_new) / (*r_dot_r_old);
        p[tid] = r[tid] + beta * p[tid];
    }
}

std::vector<LaunchConfig> launch_cg_reference(
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

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    ref_spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_ind, d_values, d_x, d_Ap);
    ref_init_r_p_kernel<<<grid, block>>>(d_b, d_Ap, d_r, d_p, N);
    
    cudaMemset(d_r_dot_r, 0, sizeof(float));
    ref_dot_kernel<<<grid, block>>>(d_r, d_r, d_r_dot_r, N);

    for (int i = 0; i < num_iters; ++i) {
        ref_spmv_csr_kernel<<<grid, block>>>(N, d_row_ptr, d_col_ind, d_values, d_p, d_Ap);
        
        cudaMemset(d_p_Ap, 0, sizeof(float));
        ref_dot_kernel<<<grid, block>>>(d_p, d_Ap, d_p_Ap, N);
        
        ref_update_x_r_kernel<<<grid, block>>>(d_x, d_r, d_p, d_Ap, d_r_dot_r, d_p_Ap, N);
        
        cudaMemset(d_r_dot_r_new, 0, sizeof(float));
        ref_dot_kernel<<<grid, block>>>(d_r, d_r, d_r_dot_r_new, N);
        
        ref_update_p_kernel<<<grid, block>>>(d_p, d_r, d_r_dot_r_new, d_r_dot_r, N);
        
        float* temp = d_r_dot_r;
        d_r_dot_r = d_r_dot_r_new;
        d_r_dot_r_new = temp;
    }

    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_r_dot_r); cudaFree(d_r_dot_r_new); cudaFree(d_p_Ap);

    return {{"ref_spmv_csr_kernel", (const void*)ref_spmv_csr_kernel, (long long)grid.x, (int)block.x, 0}};
}
