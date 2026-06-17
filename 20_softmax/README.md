# Exercise 20: Softmax

## Problem Description
**The Problem:** The Softmax function converts a vector of raw scores (logits) into a probability distribution. It involves finding the maximum value (for numerical stability), computing the exponentials, summing them, and dividing.

**Sample Input/Output:** 
- Input: `[1.0, 2.0, 3.0]`
- Output: `[0.09, 0.24, 0.66]`

**Practical Importance:** Softmax is ubiquitous in machine learning, used in the final layer for classification and within the attention mechanism of Transformers. It is heavily memory-bound due to the multiple passes typically required.

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Safe Softmax:** Always subtract the maximum value to prevent overflow during the exponential step.
- **Online Softmax:** Fused operations to compute max and sum of exponentials in a single pass using techniques like FlashAttention's online softmax.

## Objective
Implement a basic softmax kernel.
You will learn about:
- Implementing numerically stable reductions.
- Utilizing warp intrinsics for fast maximum and sum operations.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `softmax_kernel` to compute the numerically stable softmax over the last dimension of the input.
3. Implement `launch_softmax` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_softmax` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
