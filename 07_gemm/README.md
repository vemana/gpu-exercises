# Exercise 07: Matrix Multiplication (GEMM)

## Problem Description
**The Problem:** General Matrix Multiplication (GEMM) computes $C = \alpha A \times B + \beta C$. It is the core operation of linear algebra. In the naive $O(N^3)$ approach, each thread computes one element of the output matrix by taking the dot product of a row from $A$ and a column from $B$.

**Sample Input/Output:**
- Input Matrix $A$ (2x2): `[[1, 2], [3, 4]]`
- Input Matrix $B$ (2x2): `[[5, 6], [7, 8]]`
- Output Matrix $C$ (2x2): `[[19, 22], [43, 50]]`

**Practical Importance:** GEMM is arguably the most important compute kernel in modern history. The entire deep learning revolution is built upon fast matrix multiplications. Neural network layers (Dense, Convolutional, Attention mechanisms in Transformers) are mathematically reduced to massive GEMM operations. Optimized GEMM kernels are why Nvidia GPUs achieved dominance in AI, utilizing specialized hardware like Tensor Cores to reach staggering teraflops of throughput.

**Historical Anecdotes:** The BLAS (Basic Linear Algebra Subprograms) specification was published in 1979, standardizing matrix operations. GEMM (part of Level 3 BLAS) became the benchmark for supercomputer performance (the LINPACK benchmark, which decides the Top500 list). Writing a high-performance GEMM kernel on a GPU requires masterful use of shared memory tiling, register blocking, and memory coalescing.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Matrix Multiplication)
- [NVIDIA cuBLAS Documentation](https://developer.nvidia.com/cublas)
- *Anatomy of High-Performance Matrix Multiplication* by Kazushige Goto and Robert A. van de Geijn

## Newbie Guidance
**Typical CUDA Techniques:**
- **Hierarchical Tiling:** The key to a fast GEMM is maximizing the math-to-memory ratio. This requires tiling Matrix A and Matrix B into Shared Memory to share them among threads in a block, and further tiling them into Registers to share them across individual thread computations.
- **Vectorized Memory Access:** Load data using `float4` to maximize memory bandwidth utilization and reduce instruction overhead.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Tensor Cores (Ampere/Hopper/Blackwell):** Modern GEMM does not use regular CUDA cores. It uses Tensor Cores, which execute matrix-multiply-and-accumulate (MMA) instructions (like `mma.sync`) that multiply small matrices (e.g., 16x16x16) in a single clock cycle.
- **Warp Matrix Multiply Accumulate (WMMA):** The standard API for accessing Tensor Cores.
- **Tensor Memory Accelerator (TMA) and WGMMA (Hopper):** Hopper introduces Warp Group MMA (WGMMA), allowing instructions to feed directly from shared memory into Tensor Cores (bypassing registers entirely!). Combined with TMA doing asynchronous multi-dimensional fetching from global to shared memory, Hopper GEMM is entirely hardware-driven. Blackwell furthers this with even larger block sizes and native support for FP4 precision.

## Objective
Implement a general matrix multiplication (GEMM) kernel.
You will learn about:
- 2D grid and block configurations.
- Using shared memory to create a tiled matrix multiplication.
- Maximizing data reuse to overcome memory bandwidth bottlenecks.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your GEMM kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using global memory only. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `gemm_kernel` to compute the matrix product `C = A * B` where `A`, `B`, and `C` are $N \times N$ matrices.
3. Use shared memory tiling to optimize memory accesses. Load sub-tiles of `A` and `B` into shared memory, synchronize, and compute partial dot products.
4. Implement `launch_gemm` to configure the 2D grid and block dimensions, dynamically allocate shared memory (if you didn't statically allocate it), calculate occupancy metrics, and launch your kernel.
5. Make sure `launch_gemm` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
6. Compile using `make` and run `./run_test` to see if you pass the correctness tests and how your performance compares to the reference implementation.

## Typical Commands
The test suite executable `./run_test` supports various command line arguments to help you analyze and debug your kernel.

- **Help Menu**:
  ```bash
  make && ./run_test -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./run_test --size 1024
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./run_test --above 512
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./run_test --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./run_test --test_ref_kernel_only
  ```
