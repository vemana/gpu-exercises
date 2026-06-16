# Exercise 09: Radix Sort

## Problem Description
**The Problem:** Radix Sort is a non-comparative sorting algorithm. It sorts integers by processing individual digits (or bits) from least significant to most significant. In a parallel GPU context, Radix Sort is implemented by doing a stable 1-bit or multi-bit split at each step, relying heavily on parallel Scan and Stream Compaction operations.

**Sample Input/Output:**
- Input `A`: `[170, 45, 75, 90, 802, 24, 2, 66]`
- Output `C`: `[2, 24, 45, 66, 75, 90, 170, 802]`

**Practical Importance:** Sorting is a foundational computer science problem. On CPUs, `O(N log N)` comparative sorts like Quicksort or Merge Sort dominate. But on GPUs, Radix Sort is king. Because it relies on deterministic data routing (scans and scatters) rather than unpredictable branching (if statements in comparisons), it maps perfectly to the SIMD (Single Instruction, Multiple Data) architecture of GPUs. It's heavily used in computer graphics (sorting transparent objects or bounding volume hierarchies) and database indexing.

**Historical Anecdotes:** Radix sort is one of the oldest sorting algorithms, originally used in 1887 by Herman Hollerith for sorting punch cards in the US Census tabulating machines! It is incredibly poetic that a 19th-century mechanical algorithm turned out to be the most efficient way to sort data on 21st-century massively parallel GPU supercomputers.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Radix Sort)
- *Designing Efficient Sorting Algorithms for Manycore GPUs* by Nadathur Satish et al.

## Newbie Guidance
**Typical CUDA Techniques:**
- **Multi-bit Radix (Radix-4 or Radix-8):** Sorting 1 bit at a time requires 32 passes for an integer. Sorting 4 bits at a time (16 bins) requires only 8 passes. You compute a local histogram of the 4 bits in shared memory, prefix-sum it to find offsets, and scatter locally.
- **Digit Extracting:** Heavy use of bitwise operations (`>>` and `&`) to mask out the current radix digit.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp-Level Programming:** High-performance radix sorts (like CUB's implementation) heavily utilize `__ballot_sync()`, `__shfl_up_sync()`, and `__popc()` (population count) to do 1-bit or 2-bit splits entirely within the registers of a warp, completely bypassing shared memory for the local sorting phase.
- **Thread Block Clusters (Hopper):** Hopper's DSMEM allows multiple thread blocks to share their local histogram counts directly without routing through global memory, drastically speeding up the global offset computation for the bins.

## Objective
Implement a radix sort algorithm for integers.
You will learn about:
- Sorting on the GPU.
- Combining building blocks like scan and scatter to build a complex algorithm.
- Managing multiple passes over the data.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your radix sort and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using Thrust. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `radix_sort_kernel` (or a combination of kernels) to sort the array `a`, writing the sorted elements to `c`. You may want to allocate temporary buffers to ping-pong data between passes.
3. Implement `launch_radix_sort` to launch the sequence of kernels for each radix pass, calculate occupancy metrics, and return the final `LaunchMetrics`.
4. Make sure `launch_radix_sort` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
5. Compile using `make` and run `./run_test` to see if you pass the correctness tests and how your performance compares to the reference implementation.

## Typical Commands
The test suite executable `./run_test` supports various command line arguments to help you analyze and debug your kernel.

- **Help Menu**:
  ```bash
  make && ./run_test -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./run_test --size 16777216
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./run_test --above 1048576
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./run_test --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./run_test --test_ref_kernel_only
  ```
