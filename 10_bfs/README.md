# Exercise 10: Breadth-First Search (BFS)

## Problem Description
**The Problem:** Breadth-First Search (BFS) is a core graph traversal algorithm. Starting from a source node, it explores all neighboring nodes at the present depth prior to moving on to the nodes at the next depth level. In parallel, all nodes at the current frontier can explore their neighbors simultaneously.

**Sample Input/Output:**
- Graph Edges: `0->1, 0->2, 1->3, 2->3, 3->4`
- Source Node: `0`
- Output Distances: `[0, 1, 1, 2, 3]`

**Practical Importance:** Graph processing is crucial for analyzing relationships in massive datasets. BFS is the foundation for shortest path algorithms (like Dijkstra's), network routing protocols, and social network analysis (e.g., finding the "degrees of separation" between two users). In VLSI design, it's used for circuit layout and testing.

**Historical Anecdotes:** BFS was invented in the late 1950s by Edward F. Moore to find the shortest path out of a maze. Parallelizing BFS on GPUs was long considered a "holy grail" because graph memory accesses are notoriously irregular and sparse, causing terrible cache performance. Breaking the "GTEPS" (Giga-Traversed Edges Per Second) barrier on GPUs required immense creativity in managing frontiers.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Graph Search)
- [Graph500 Benchmark](https://graph500.org/)
- *Accelerating large graph algorithms on the GPU using CUDA* by P. Harish and P. J. Narayanan


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:**
- **Frontier Queues:** BFS requires keeping track of the "frontier" (nodes currently being visited). You need to maintain two queues: `current_frontier` and `next_frontier`. Threads pull nodes from the current queue, examine neighbors, and use `atomicAdd` to push unvisited neighbors onto the next queue.
- **CSR Graph Format:** Storing the graph in Compressed Sparse Row (CSR) format ensures that looking up a node's neighbors requires contiguous memory reads.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Cooperative Groups:** Exploring nodes often results in massive warp divergence (some nodes have 1 neighbor, some have 10,000). Cooperative Groups allow you to dynamically assign entire warps or blocks to process a single high-degree node, load-balancing the work.
- **L2 Cache & DSMEM (Hopper):** Graph traversal is deeply limited by random memory access. Utilizing Hopper's Thread Block Clusters to keep related graph partitions loaded in Distributed Shared Memory (DSMEM) reduces L2 cache thrashing.

## Objective
Implement a Breadth-First Search (BFS) graph traversal algorithm on the GPU.
You will learn about:
- Operating on graph structures using Compressed Sparse Row (CSR) format.
- Multi-kernel level-synchronous execution.
- Managing control flow and iterative updates between CPU and GPU.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your BFS kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using a simple level-synchronous approach. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `bfs_kernel` to perform a single iteration (level) of BFS. It should look at nodes at the current level, check their neighbors, and update the neighbors' distances if they haven't been visited, signaling that a change occurred.
3. Implement `launch_bfs` to initialize the source node's distance, and iteratively launch your kernel until no new distances are updated.
4. Make sure `launch_bfs` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
5. Compile using `make` and run `./bin/run_test.sh` to evaluate your correctness and performance, and `./bin/run_profiler.sh` to identify bottlenecks.

## Typical Commands
The test suite executables in `./bin/` support various command line arguments to help you analyze and debug your kernel. You can use `./bin/run_test.sh` to run the kernel normally, or `./bin/run_profiler.sh` to run it under Nsight Compute for detailed profiling.

- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  make && ./bin/run_profiler.sh -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./bin/run_test.sh --size 16777216
  ```
- **Profile Memory Metrics** (runs Nsight Compute with predefined metrics for memory-bound kernels):
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 16777216
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./bin/run_test.sh --above 1048576
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./bin/run_test.sh --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./bin/run_test.sh --test_ref_kernel_only
  ```
