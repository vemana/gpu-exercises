# Exercise 22: Depthwise Convolution 2D

## Problem Description
**The Problem:** A depthwise convolution applies a single spatial filter to each input channel independently, rather than combining all channels like a standard convolution.

**Sample Input/Output:** 
- Input `[C=2, H=3, W=3]`:
  ```
  Channel 0: [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
  Channel 1: [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
  ```
- Filters `[C=2, K=2, K=2]`:
  ```
  Filter 0: [[1, 0], [0, 1]]
  Filter 1: [[1, 1], [1, 1]]
  ```
- Output `[C=2, H=2, W=2]`:
  ```
  Channel 0: [[2, 2], [2, 2]]
  Channel 1: [[8, 8], [8, 8]]
  ```

**Practical Importance:** Depthwise convolutions drastically reduce the number of parameters and computations compared to standard convolutions. They are the core building block of efficient mobile architectures like MobileNet.

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Shared Memory Caching:** Load a tile of the input image and the filter into shared memory to reduce redundant global memory loads for overlapping patches.

## Objective
Implement a basic depthwise 2D convolution kernel.
You will learn about:
- Exploiting data reuse with shared memory in a stencil-like operation.
- Channel-independent spatial filtering.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `depthwise_conv2d_kernel` to apply a spatial filter to each channel of the input independently.
3. Implement `launch_depthwise_conv2d` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_depthwise_conv2d` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
