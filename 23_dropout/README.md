# Exercise 23: Dropout

## Problem Description
**The Problem:** Dropout is a regularization technique that randomly zeroes out elements of the input tensor with a probability $p$, and scales the remaining elements by $1 / (1 - p)$ to maintain the expected value.

**Sample Input/Output:** 
- Input: `[1.0, 1.0, 1.0, 1.0]`, $p=0.5$.
- Output might be: `[2.0, 0.0, 2.0, 0.0]`.

**Practical Importance:** Essential for preventing overfitting in large neural networks during training. Implementing dropout efficiently on the GPU requires fast pseudorandom number generation (PRNG).

## Newbie Guidance
**Typical CUDA Techniques:** 
- **GPU PRNG:** Generating random numbers on the GPU. We use a simple deterministic PCG Hash to avoid state-keeping overhead and ensure perfectly reproducible testing.
- **Vectorized Loads/Writes:** Since this is element-wise, using `float4` to process 4 elements per thread can boost bandwidth.

## Objective
Implement a basic dropout kernel.
You will learn about:
- Pseudorandom number generation in a CUDA kernel.
- Element-wise scaling and masking.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `dropout_kernel` to randomly zero out elements using a deterministic hash and scale the rest.
3. Implement `launch_dropout` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_dropout` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
