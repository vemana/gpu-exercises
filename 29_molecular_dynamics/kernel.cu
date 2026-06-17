
// Note: HPC applications typically use double precision.
// We use float here for consistency across the repository exercises.

#include "kernel.h"

#include <cmath>
#include <vector>

#include "../utils/framework.h"

__global__ void build_cells_kernel(const float2* pos, int* cell_starts, int* particle_next, 
                                   int N, float cutoff_radius, int grid_dim) {
    // TODO: Implement your kernel here
}

__global__ void compute_forces_kernel(const float2* pos, float2* forces, 
                                      const int* cell_starts, const int* particle_next,
                                      int N, float cutoff_radius, int grid_dim) {
    // TODO: Implement your kernel here
}

std::vector<LaunchConfig> launch_md(
    const float2* d_pos, float2* d_forces, 
    int N, float domain_size, float cutoff_radius) {
    
    int grid_dim = ceil(domain_size / cutoff_radius);
    int num_cells = grid_dim * grid_dim;
    
    int *d_cell_starts, *d_particle_next;
    cudaMalloc(&d_cell_starts, num_cells * sizeof(int));
    cudaMalloc(&d_particle_next, N * sizeof(int));
    
    cudaMemset(d_cell_starts, -1, num_cells * sizeof(int));
    
    // TODO: Define grid and block dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    build_cells_kernel<<<grid, block>>>(d_pos, d_cell_starts, d_particle_next, N, cutoff_radius, grid_dim);
    compute_forces_kernel<<<grid, block>>>(d_pos, d_forces, d_cell_starts, d_particle_next, N, cutoff_radius, grid_dim);
    
    cudaFree(d_cell_starts);
    cudaFree(d_particle_next);

    return {{"compute_forces_kernel", (const void*)compute_forces_kernel, (long long)grid.x, (int)block.x, 0}};
}
