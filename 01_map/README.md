# Exercise 01: Vector Addition (Map)

## Problem Description
**The Problem:** The Map pattern applies a given function independently to each element in a collection. In this exercise, we focus on the simplest form of Map: Vector Addition ($C_i = A_i + B_i$). Since each element's computation does not depend on any other element, this problem is "embarrassingly parallel."

**Sample Input/Output:** 
- Input `A`: `[1.0, 2.0, 3.0, 4.0]`
- Input `B`: `[5.0, 6.0, 7.0, 8.0]`
- Output `C`: `[6.0, 8.0, 10.0, 12.0]`

**Practical Importance:** The Map pattern is the absolute backbone of data-parallel computing. From adjusting pixel brightness in an image filter (where every pixel is mapped to a new value) to activation functions in deep neural networks like ReLUs, mapping is everywhere. In frontier fields like large language models (LLMs), operations like layer normalization and point-wise feed-forward networks scale to billions of parameters, relying entirely on highly optimized map operations across thousands of GPU cores.

**Historical Anecdotes:** The term "Map" dates back to LISP in the late 1950s. Decades later, Google popularized the "MapReduce" framework for processing massive datasets across distributed clusters. With the advent of general-purpose GPU computing (GPGPU), the Map paradigm became the natural starting point for translating CPU loops into GPU kernels.

**References:** 
- *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu (Chapter 2: Data Parallel Computing)
- Google's original 2004 MapReduce paper: [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/pub62/)

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Memory Coalescing:** The most critical optimization here. Ensure that adjacent threads in a warp access adjacent memory addresses in global memory. If `A` and `B` are accessed contiguously, the hardware can combine multiple memory requests into a single transaction.
- **Grid-Stride Loops:** Instead of assuming the grid size exactly matches the array size, use a loop (`for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)`) to allow your kernel to process arrays of any size seamlessly.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Asynchronous Data Copies:** For simple memory-bound kernels like Map, overlapping memory transfers and computation using CUDA Streams is vital. On Ampere and Hopper architectures, hardware-accelerated async copies (e.g., `cudaMemcpyAsync`) and the new Tensor Memory Accelerator (TMA) on Hopper can drastically improve PCIe/NVLink data movement efficiency.

## Objective
Implement a basic vector addition kernel.
You will learn about:
- Launching a basic CUDA kernel.
- Calculating thread IDs and accessing data.
- Basic memory management (handled by framework).

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your map kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `map_kernel` to add two vectors `a` and `b` and store the result in `c`.
3. Implement `launch_map` to configure the grid and block dimensions, calculate occupancy metrics, and launch your kernel.
4. Make sure `launch_map` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
5. Compile using `make` and run `./run_test` to see if you pass the correctness tests and how your performance compares to the reference implementation.

## Typical Commands
The test suite executable `./run_test` supports various command line arguments to help you analyze and debug your kernel.

- **Help Menu**:
  ```bash
  make && ./run_test -h
  ```
- **Test a Specific Size** (tests the predefined size closest to the given value, preferring the higher value in a tie):
  ```bash
  make && ./run_test --size 16777216
  ```
- **Test Sizes Above a Threshold**:
  ```bash
  make && ./run_test --above 1048576
  ```
- **Enable Verbose Tracing** (prints detailed setup and launch trace logs):
  ```bash
  make && ./run_test --verbose
  ```
- **Run the Reference Kernel Only** (verifies correctness of the reference kernel without testing or benchmarking your implementation):
  ```bash
  make && ./run_test --test_ref_kernel_only
  ```
