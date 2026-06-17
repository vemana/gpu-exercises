# Exercise 17: Max Pooling 2D

## Problem Description
**The Problem:** Max pooling reduces the spatial dimensions of an image or feature map by taking the maximum value over a sliding window.

**Sample Input/Output:** 
- Window: 2x2, Stride: 2
- Input `4x4`:
  ```
  [[1, 2, 3, 4],
   [5, 6, 7, 8],
   [9, 10, 11, 12],
   [13, 14, 15, 16]]
  ```
- Output `2x2`:
  ```
  [[6, 8],
   [14, 16]]
  ```

**Practical Importance:** Pooling is a foundational operation in Convolutional Neural Networks (CNNs). It provides translation invariance and reduces computational cost for subsequent layers.

## Newbie Guidance
**Typical CUDA Techniques:** 
- **2D Grid Mapping:** Map 2D output coordinates to CUDA block and thread indices.
- **Boundary Checking:** Handle edge cases where the pooling window extends beyond the input image boundaries.

## Objective
Implement a basic max pooling 2D kernel.
You will learn about:
- Mapping 2D grid/block coordinates to multi-dimensional tensor indices.
- Implementing spatial reductions.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `max_pool2d_kernel` to perform a 2D max pooling operation on a multi-channel input image.
3. Implement `launch_max_pool2d` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_max_pool2d` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
