#include "reference_kernel.h"
#include <cuda_runtime.h>
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void reference_bfs_frontier_kernel(const int* current_frontier, int frontier_size,
                                             const int* row_offsets, const int* col_indices, 
                                             int* distances, 
                                             int* next_frontier, int* next_frontier_size, 
                                             int current_level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int u = current_frontier[tid];
        int start_edge = row_offsets[u];
        int end_edge = row_offsets[u + 1];
        
        for (int i = start_edge; i < end_edge; ++i) {
            int v = col_indices[i];
            // Try to claim the node for the next level
            if (atomicCAS(&distances[v], -1, current_level + 1) == -1) {
                // If successfully claimed, add to next frontier
                int idx = atomicAdd(next_frontier_size, 1);
                next_frontier[idx] = v;
            }
        }
    }
}

LaunchMetrics launch_reference_bfs(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int num_edges, int source_node) {
    global_tracer.trace("Entering launch_reference_bfs (Frontier Queue Optimized)");
    
    int* d_current_frontier;
    int* d_next_frontier;
    int* d_next_frontier_size;
    
    cudaMalloc(&d_current_frontier, num_nodes * sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes * sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));
    
    int initial_frontier_size = 1;
    cudaMemcpy(d_current_frontier, &source_node, sizeof(int), cudaMemcpyHostToDevice);
    
    int zero = 0;
    cudaMemcpy(distances + source_node, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    int current_level = 0;
    int frontier_size = 1;
    int threadsPerBlock = 256;
    int maxBlocksPerGrid = 0;
    
    while (frontier_size > 0) {
        cudaMemset(d_next_frontier_size, 0, sizeof(int));
        
        int blocksPerGrid = (frontier_size + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > maxBlocksPerGrid) maxBlocksPerGrid = blocksPerGrid;
        
        reference_bfs_frontier_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_current_frontier, frontier_size,
            row_offsets, col_indices,
            distances,
            d_next_frontier, d_next_frontier_size,
            current_level
        );
        
        cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Swap frontiers
        int* temp = d_current_frontier;
        d_current_frontier = d_next_frontier;
        d_next_frontier = temp;
        
        current_level++;
    }
    
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);
    
    OccupancyMetrics occ = calculate_occupancy((const void*)reference_bfs_frontier_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Exiting launch_reference_bfs");
    return {maxBlocksPerGrid, occ};
}
