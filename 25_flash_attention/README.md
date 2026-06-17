# Exercise 25: Flash Attention (Simplified)

## Problem Description
**The Problem:** Standard Attention computes $O = \text{softmax}(Q K^T / \sqrt{d}) V$. The intermediate attention matrix $S = Q K^T$ is of size $N \times N$, which causes massive memory bottlenecks for long sequences. Flash Attention computes the exact same result without materializing the $N \times N$ matrix in global memory.

**Sample Input/Output:** 
- Q = K = `[[1.0, 0.0], [0.0, 1.0]]`
- V = `[[10.0, 20.0], [30.0, 40.0]]`
- $S = Q K^T$ = `[[1.0, 0.0], [0.0, 1.0]]`
- $O = \text{Softmax}(S) V$ = `[[15.4, 25.4], [24.6, 34.6]]`

**Practical Importance:** Flash Attention revolutionized Transformer context lengths. By fusing the matrix multiplications and the softmax, and tiling the computation to fit in SRAM (shared memory), it changes attention from memory-bandwidth-bound to compute-bound.


> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Kernel Fusion:** Combine GEMM, Softmax, and another GEMM into a single kernel.
- **Tiling:** Load blocks of Q, K, V into shared memory to compute partial attention outputs.
- **Online Softmax:** Keep track of running maximums and denominator sums to incrementally compute softmax.

## Objective
Implement a basic flash attention kernel.
You will learn about:
- Advanced kernel fusion and tiling.
- Online softmax computation.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `flash_attention_kernel` to compute the attention output $O$ without materializing the full $N \times N$ attention matrix.
3. Implement `launch_flash_attention` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_flash_attention` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
