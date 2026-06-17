# Exercise 21: Im2Col

## Problem Description
**The Problem:** Image-to-Column (Im2Col) is a memory reshaping operation that transforms overlapping local image regions (patches) into flattened columns.

**Sample Input/Output:** 
- Input Image `[C=1, H=3, W=3]`:
  ```
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]
  ```
- Kernel `2x2`, Stride `1`.
- Output Matrix `[C*K*K=4, Out_H*Out_W=4]`:
  ```
  [[1, 2, 4, 5],
   [2, 3, 5, 6],
   [4, 5, 7, 8],
   [5, 6, 8, 9]]
  ```

**Practical Importance:** Instead of writing complex nested loops for 2D convolutions, Im2Col allows convolution to be computed as a single large, highly-optimized General Matrix Multiplication (GEMM). This is how most deep learning frameworks implement convolutions under the hood.


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Index Arithmetic:** Complex mapping from the flattened 1D thread ID back to the 3D input tensor and 2D output matrix.
- **Memory Bandwidth:** The output is much larger than the input. Maximizing global memory write throughput is essential.

## Objective
Implement a basic im2col kernel.
You will learn about:
- Complex tensor index transformations.
- Understanding the memory layout mechanics behind deep learning convolutions.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `im2col_kernel` to extract image patches and flatten them into a column-wise matrix.
3. Implement `launch_im2col` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_im2col` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
