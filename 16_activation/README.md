# Exercise 16: Activation Functions (SiLU)

## Problem Description
**The Problem:** Activation functions introduce non-linearity into neural networks. In this exercise, we implement the SiLU (Sigmoid Linear Unit) or Swish activation function: $f(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$.

**Sample Input/Output:** 
- Input: `[0.0, 1.0, -1.0]`
- Output: `[0.0, 0.731, -0.268]`

**Practical Importance:** SiLU is widely used in modern LLMs like LLaMA and Mistral. Because it is an element-wise operation, it is heavily memory-bound. Optimizing element-wise operations is crucial because they occupy a non-trivial fraction of total execution time in deep networks.


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Memory Coalescing:** Ensure adjacent threads read/write adjacent elements.
- **Math Intrinsics:** Use fast math functions like `__expf()` or standard `expf()` to optimize transcendental math operations.

## Objective
Implement a basic activation kernel.
You will learn about:
- Applying element-wise mathematical functions.
- Utilizing CUDA math intrinsics.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `activation_kernel` to apply the SiLU activation function to each element of the input tensor.
3. Implement `launch_activation` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_activation` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
