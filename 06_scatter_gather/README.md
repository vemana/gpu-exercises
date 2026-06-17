# Exercise 06: Scatter/Gather

## Problem Description
**The Problem:** Scatter and Gather are fundamental memory access patterns. **Gather** reads from arbitrary locations (e.g., `A[indices[i]]`) and writes to a contiguous output. **Scatter** reads from a contiguous input and writes to arbitrary locations (e.g., `C[indices[i]] = A[i]`). These patterns test the memory subsystem's ability to handle non-coalesced, random accesses.

**Sample Input/Output (Gather):**
- Input Data `A`: `[10, 20, 30, 40, 50]`
- Indices: `[4, 0, 2, 2, 1]`
- Output `C`: `[50, 10, 30, 30, 20]`

**Practical Importance:** Scatter/Gather operations bridge the gap between dense arrays and sparse data structures. They are essential for sparse matrix-vector multiplication (SpMV), graph algorithms (like PageRank), and routing data in networking. In machine learning, embedding lookups (used heavily in recommendation systems and LLMs) are essentially massive Gather operations, pulling specific vectors from a giant continuous weight matrix based on token IDs.

**Historical Anecdotes:** The distinction between Scatter and Gather is deeply tied to hardware. Historically, vector supercomputers (like the Cray-1 in 1976) introduced hardware support for Scatter/Gather to vectorize loops with indirect addressing. In early GPUs, Scatter was notoriously unsupported or extremely slow, forcing programmers to creatively re-architect their algorithms into Gathers.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Sparse Matrix Computation)
- [Wikipedia: Gather-scatter (vector addressing)](https://en.wikipedia.org/wiki/Gather-scatter_(vector_addressing))


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:**
- **L1/L2 Caching:** Gather operations rely entirely on the GPU's cache hierarchy to perform well, as accesses are non-coalesced. If the indices exhibit temporal or spatial locality, the L2 cache is your best friend.
- **Data Layout:** The ultimate solution to poor scatter/gather performance is often avoiding it by restructuring the data layout from Arrays of Structures (AoS) to Structures of Arrays (SoA).

**Modern CUDA Primitives (Ampere & Beyond):**
- **L2 Cache Residency Controls:** Ampere and newer architectures allow you to specifically "pin" a portion of the data (e.g., a heavily accessed lookup table) into the L2 cache, preventing it from being evicted. This drastically accelerates random gather reads.
- **Hopper DPX Instructions:** Hopper introduces dynamic programming instructions that can optimize irregular data access patterns, though memory subsystem mastery remains the primary focus.

## Objective
Implement a gather operation to reorganize elements of an array.
You will learn about:
- Non-coalesced memory accesses and their impact on memory bandwidth.
- Translating indices to gather sparse or randomly ordered data into a dense sequence.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your gather kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using global memory. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `gather_kernel` to gather elements from `source` using the permutation `indices`, writing the results to `dest` such that `dest[i] = source[indices[i]]`.
3. Implement `launch_gather` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_gather` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
