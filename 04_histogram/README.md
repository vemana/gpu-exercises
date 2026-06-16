# Exercise 04: Histogram

## Problem Description
**The Problem:** A Histogram takes an array of values and counts the frequency of each value (or range of values) into discrete bins. Since multiple threads might try to increment the same bin counter simultaneously, this requires atomic operations to avoid race conditions.

**Sample Input/Output:**
- Input `A`: `[1, 0, 2, 2, 1, 1, 3, 0]`
- Num Bins: `4`
- Output Bins: `[2, 3, 2, 1]`

**Practical Importance:** Histograms are crucial for data distribution analysis. In computer vision and computational photography, histograms are used to adjust image contrast (histogram equalization). In natural language processing, word frequencies are essentially histograms. In databases, query optimizers rely heavily on data histograms to estimate the cost of different query plans and choose the fastest execution route.

**Historical Anecdotes:** The word "histogram" is derived from the Greek *histos* (anything set upright) and *gramma* (drawing). It was introduced by Karl Pearson in 1891. In GPU computing, histograms were a major motivation for hardware engineers to add fast atomic operations to shared memory, dramatically shifting GPUs from pure graphics pipelines to general-purpose data processors.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Histogram Computation)
- NVIDIA CUDA C++ Programming Guide (Section on Atomic Functions)

## Newbie Guidance
**Typical CUDA Techniques:**
- **Privatized Histograms:** Instead of thousands of threads hammering the same global memory bins using `atomicAdd` (which causes terrible serialization), have each thread block compute a local "private" histogram in its shared memory first. Then, safely atomic-add the shared memory bins into the global memory bins at the end.
- **Memory Coalescing:** Ensure that threads read the input data contiguously, even if they write to scattered histogram bins.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Hardware-Accelerated Atomics:** Ampere and Hopper have vastly improved throughput for atomic operations to both global and shared memory. However, contention is still the enemy. 
- **Thread Block Clusters (Hopper):** If bins exceed shared memory limits, Hopper allows multiple blocks in a cluster to share a larger pool of Distributed Shared Memory (DSMEM). This means you can create larger "privatized" histograms across a cluster of blocks before touching global memory.

## Objective
Implement a kernel to compute the histogram of an array of integers.
You will learn about:
- Using `atomicAdd` to safely update global memory from multiple threads.
- Shared memory optimization (using `atomicAdd` on shared memory first before committing to global memory) to reduce global memory contention.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your histogram kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using global memory atomics. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `histogram_kernel` to compute the bin frequencies of the `a` array into `bins`. The number of bins is given by `num_bins`.
3. Implement `launch_histogram` to configure the grid and block dimensions, dynamically allocate shared memory (if you choose to use it), calculate occupancy metrics, and launch your kernel.
4. Make sure `launch_histogram` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
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
