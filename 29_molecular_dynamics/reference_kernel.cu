

#include "reference_kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void ref_build_cells_kernel(const float2* pos, int* cell_starts, int* particle_next, 
                                       int N, float cutoff_radius, int grid_dim) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 p = pos[i];
        int cx = min(max((int)(p.x / cutoff_radius), 0), grid_dim - 1);
        int cy = min(max((int)(p.y / cutoff_radius), 0), grid_dim - 1);
        int cell_idx = cy * grid_dim + cx;
        
        int old_head = atomicExch(&cell_starts[cell_idx], i);
        particle_next[i] = old_head;
    }
}

__global__ void ref_compute_forces_kernel(const float2* pos, float2* forces, 
                                          const int* cell_starts, const int* particle_next,
                                          int N, float cutoff_radius, int grid_dim) {
    // REFERENCE IMPLEMENTATION:
    // This kernel is provided for correctness and reference.
    // It processes the data according to the mathematical definition of the algorithm.
    // Pay attention to the thread indexing and boundary checks.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 p_i = pos[i];
        float2 f_i = make_float2(0.0f, 0.0f);
        
        int cx = min(max((int)(p_i.x / cutoff_radius), 0), grid_dim - 1);
        int cy = min(max((int)(p_i.y / cutoff_radius), 0), grid_dim - 1);
        
        float cutoff_sq = cutoff_radius * cutoff_radius;
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                
                if (nx >= 0 && nx < grid_dim && ny >= 0 && ny < grid_dim) {
                    int cell_idx = ny * grid_dim + nx;
                    int j = cell_starts[cell_idx];
                    
                    while (j != -1) {
                        if (i != j) {
                            float2 p_j = pos[j];
                            float dx_pos = p_i.x - p_j.x;
                            float dy_pos = p_i.y - p_j.y;
                            float r2 = dx_pos*dx_pos + dy_pos*dy_pos;
                            
                            if (r2 < cutoff_sq && r2 > 0.001f) {
                                float r2_inv = 1.0f / r2;
                                float r6_inv = r2_inv * r2_inv * r2_inv;
                                float f_mag = 24.0f * r6_inv * (2.0f * r6_inv - 1.0f) * r2_inv;
                                
                                f_i.x += f_mag * dx_pos;
                                f_i.y += f_mag * dy_pos;
                            }
                        }
                        j = particle_next[j];
                    }
                }
            }
        }
        forces[i] = f_i;
    }
}

std::vector<LaunchConfig> launch_md_reference(
    const float2* d_pos, float2* d_forces, 
    int N, float domain_size, float cutoff_radius) {
    
    int grid_dim = ceil(domain_size / cutoff_radius);
    int num_cells = grid_dim * grid_dim;
    
    int *d_cell_starts, *d_particle_next;
    cudaMalloc(&d_cell_starts, num_cells * sizeof(int));
    cudaMalloc(&d_particle_next, N * sizeof(int));
    
    cudaMemset(d_cell_starts, -1, num_cells * sizeof(int));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    ref_build_cells_kernel<<<grid, block>>>(d_pos, d_cell_starts, d_particle_next, N, cutoff_radius, grid_dim);
    ref_compute_forces_kernel<<<grid, block>>>(d_pos, d_forces, d_cell_starts, d_particle_next, N, cutoff_radius, grid_dim);
    
    cudaFree(d_cell_starts);
    cudaFree(d_particle_next);

    return {{"ref_compute_forces_kernel", (const void*)ref_compute_forces_kernel, (long long)grid.x, (int)block.x, 0}};
}
