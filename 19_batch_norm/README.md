# Exercise 19: Batch Normalization

## Problem Description
**The Problem:** Batch Normalization standardizes the activations of a layer across the batch and spatial dimensions for each channel.

**Sample Input/Output:** 
- Input `[N=2, C=1, H=1, W=2]`:
  ```
  [[[[1.0, 2.0]]], [[[3.0, 4.0]]]]
  ```
- Output (normalized over N, H, W for C=0):
  ```
  [[[[-1.34, -0.44]]], [[[0.44, 1.34]]]]
  ```

**Practical Importance:** BatchNorm is standard in vision models like ResNet. It requires reducing a massive number of elements (batch $	imes$ spatial) into a single mean and variance per channel, testing global reduction capabilities.


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Grid-Level Reductions:** Requires synchronizing across multiple blocks or using atomic operations, since one block usually isn't enough to process the entire batch and spatial dimension for a single channel.

## Objective
Implement a basic batch normalization kernel.
You will learn about:
- Global memory atomic additions or multi-kernel reductions.
- Handling non-contiguous memory access patterns efficiently.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `batch_norm_kernel` to normalize the input tensor across the batch and spatial dimensions per channel.
3. Implement `launch_batch_norm` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_batch_norm` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
