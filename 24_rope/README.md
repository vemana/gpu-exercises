# Exercise 24: Rotary Positional Embeddings (RoPE)

## Problem Description
**The Problem:** RoPE encodes positional information directly into the Query and Key vectors in attention mechanisms by rotating pairs of features in the hidden dimension.

**Sample Input/Output:** 
- Input (1 token at pos $m=1$, dim 2): `[x0=1.0, x1=2.0]`
- Rotation angle $\theta = 1 \times 10000^{-0/2} = 1.0$ rad
- Output: `[x0*cos(1.0) - x1*sin(1.0), x0*sin(1.0) + x1*cos(1.0)]`
  $= [-1.14, 1.92]$

**Practical Importance:** RoPE is the standard positional encoding used in modern LLMs (LLaMA, PaLM, etc.). It enables better length extrapolation compared to absolute positional embeddings.

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Trigonometric Math:** Extensive use of `sin` and `cos`. Use CUDA intrinsics (`__sincosf`) to compute both simultaneously for better performance.

## Objective
Implement a basic rotary positional embedding kernel.
You will learn about:
- Modifying specific pairs of elements within a tensor.
- Optimizing trigonometric operations.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `rope_kernel` to apply rotary positional embeddings to the input sequence.
3. Implement `launch_rope` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_rope` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
