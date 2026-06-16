# Exercise 13: Merge Sort

## Problem Description
**The Problem:** Merge Sort is a divide-and-conquer algorithm that recursively splits an array in half, sorts the halves, and then merges them back together. In a GPU context, the recursive splitting is usually abandoned in favor of an iterative bottom-up approach, where threads merge small pairs of elements, then larger sorted runs, until the whole array is sorted. 

**Sample Input/Output:**
- Input `A`: `[38, 27, 43, 3, 9, 82, 10]`
- Merging pairs: `[27, 38] [3, 43] [9, 82] [10]`
- Merging fours: `[3, 27, 38, 43] [9, 10, 82]`
- Output `C`: `[3, 9, 10, 27, 38, 43, 82]`

**Practical Importance:** While Radix Sort dominates for integers, Merge Sort is highly competitive for sorting arbitrary data types (like floating point numbers, strings, or complex structs) that require a comparison operator. It is completely stable and its memory access patterns (reading from two arrays, writing to a third) can be highly coalesced if the split points are carefully chosen using binary search algorithms (like the Merge Path algorithm).

**Historical Anecdotes:** Merge sort was invented by John von Neumann in 1945 for the EDVAC computer. When porting it to GPUs, a major breakthrough was the "Merge Path" algorithm developed by Odeh et al., which provided a perfectly load-balanced way to assign work to GPU threads when merging two arbitrary sorted arrays.

**References:**
- *Merge Path - A Visually Intuitive Approach to Parallel Merging* by Oded Green et al.
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Merge Sort)

## Newbie Guidance
**Typical CUDA Techniques:**
- **Merge Path Algorithm:** To parallelize merging two sorted arrays, you can't just split the arrays randomly. You must use binary search to find the "Merge Path" — the exact indices in Array A and Array B that divide the elements perfectly into equal chunks for each thread block.
- **Shared Memory Ping-Pong:** Once a thread block has identified its chunk of A and B, it loads them into shared memory and performs a localized sequential (or warp-parallel) merge into a temporary shared memory buffer before writing to global memory.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Cooperative Groups:** Managing the multi-stage nature of Merge Sort (merging size 2, then 4, then 8... up to N) usually requires many kernel launches. With Cooperative Groups' `grid_sync()`, you can synchronize the entire GPU in a single kernel launch, maintaining cache warmth across merge passes.
- **Distributed Shared Memory (Hopper):** Thread Block Clusters can share their sorted chunks directly via DSMEM, allowing larger contiguous sequences to be merged without spilling to the L2 cache.

## Objective
Implement a parallel merge sort algorithm.
You will learn about:
- Divide and conquer algorithms on the GPU.
- Finding split points using binary search (e.g., merge path algorithm) for load balancing.
- Hierarchical merging.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your merge sort kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using Thrust. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `merge_sort_kernel` (or a series of kernels) to sort the array `a` into `c`. You'll likely need ping-pong buffers as you iteratively merge larger and larger sorted runs.
3. Implement `launch_merge_sort` to configure the grid and block dimensions, calculate occupancy metrics, and launch your kernels.
4. Make sure `launch_merge_sort` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
5. Compile using `make` and run `./bin/run_test` to see if you pass the correctness tests and how your performance compares to the reference implementation.

## Typical Commands
The test suite executable `./bin/run_test` supports various command line arguments to help you analyze and debug your kernel.

- **Help Menu**:
  ```bash
  make && ./bin/run_test -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./bin/run_test --size 16777216
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./bin/run_test --above 1048576
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./bin/run_test --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./bin/run_test --test_ref_kernel_only
  ```
