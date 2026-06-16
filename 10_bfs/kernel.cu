#include "kernel.h"
#include "../utils/utils.h"
#include "../utils/tracer.h"

__global__ void bfs_kernel(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int current_level, bool* changed) {
    // TODO: Implement one level of BFS
}

LaunchMetrics launch_bfs(const int* row_offsets, const int* col_indices, int* distances, int num_nodes, int num_edges, int source_node) {
    global_tracer.trace("Entering launch_bfs (Student)");
    
    // TODO: Set distance of source_node to 0 (you may need a separate small kernel or cudaMemcpy)
    
    // TODO: Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
    
    // TODO: Loop until no changes are made
    
    OccupancyMetrics occ = calculate_occupancy((const void*)bfs_kernel, threadsPerBlock, 0);
    
    global_tracer.trace("Exiting launch_bfs (Student)");
    return {blocksPerGrid, occ};
}
