# Exercise 18: Layer Normalization

## Problem Description
**The Problem:** Layer Normalization standardizes the inputs to a layer across the hidden dimension, independently for each sequence/batch element. It computes the mean and variance for each row and normalizes it.

**Sample Input/Output:** 
- Input `[Batch=2, Hidden_Dim=3]`:
  ```
  [[1.0, 2.0, 3.0],
   [4.0, 5.0, 6.0]]
  ```
- Output normalized (row mean=0, var=1):
  ```
  [[-1.22, 0.0, 1.22],
   [-1.22, 0.0, 1.22]]
  ```

**Practical Importance:** LayerNorm is a vital component of Transformer models. It stabilizes training by preventing extreme activations. Since it requires reducing across the hidden dimension, it is typically memory-bound but requires intra-block communication.

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Block/Warp Reductions:** Use shared memory or warp shuffle instructions (e.g., `__shfl_down_sync`) to efficiently sum elements across a block.
- **Multiple Passes vs Single Pass:** You can compute mean and variance in two passes or using Welford's algorithm in a single pass.

## Objective
Implement a basic layer normalization kernel.
You will learn about:
- Performing parallel reductions across a specific tensor dimension.
- Sharing data between threads within a block.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `layer_norm_kernel` to normalize each row of the input tensor independently.
3. Implement `launch_layer_norm` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_layer_norm` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
