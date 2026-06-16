# Exercise 05: 1D Stencil

## Problem Description
**The Problem:** A Stencil operation updates elements in an array based on a fixed pattern (the stencil) of neighboring elements. It is a localized, sliding-window operation often applied to grids or matrices, representing physical diffusion or smoothing.

**Sample Input/Output (1D, 3-point stencil averaging):**
- Input `A`: `[10, 20, 30, 40, 50]`
- Stencil Weights: `[0.25, 0.5, 0.25]`
- Output `C` (inner elements): `[..., 20, 30, 40, ...]`

**Practical Importance:** Stencil computations are the workhorses of scientific simulations. They are used to solve partial differential equations (PDEs) like the heat equation, fluid dynamics (Navier-Stokes), and weather forecasting models. In image processing, stencils are used for blur, edge detection, and sharpening filters. Because stencils read the same neighborhood repeatedly, they are prime candidates for shared memory optimization (tiling) to save memory bandwidth.

**Historical Anecdotes:** Stencil algorithms have roots in the finite difference method, developed in the early 20th century by Lewis Fry Richardson to predict weather by hand! Richardson calculated that he would need 64,000 human "computers" working together in a giant hall to predict the weather as fast as it happened. Today, a single GPU can perform billions of stencil operations per second.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Stencil Computation)
- [Wikipedia: Stencil Code](https://en.wikipedia.org/wiki/Stencil_code)

## Newbie Guidance
**Typical CUDA Techniques:**
- **Shared Memory Tiling + Halos:** Stencil operations exhibit massive data reuse (overlapping windows). Load a "tile" of the input array into shared memory, including the necessary boundary regions (the "halo"). Synchronize (`__syncthreads()`), and then let all threads in the block compute their stencil from the ultra-fast shared memory.
- **Register Tiling:** If the stencil footprint is small, loading values directly into registers and shuffling them between threads can be even faster than shared memory.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Asynchronous Copy (`cuda::memcpy_async`):** Ampere introduced hardware-accelerated async copies from global to shared memory. You can initiate the load of the tile and halo into shared memory and have the GPU fetch it in the background while threads do other work, completely bypassing the register file.
- **Tensor Memory Accelerator (TMA) (Hopper):** Hopper takes this further. TMA allows you to specify a multi-dimensional tensor block (e.g., a 2D or 3D tile with its halo). The hardware will autonomously fetch the exact bounded region into shared memory asynchronously, automatically handling boundary conditions without you needing to write complex if-statements!

## Objective
Implement a 1D stencil computation.
You will learn about:
- Loading halo/ghost regions into shared memory.
- Synchronizing threads within a block using `__syncthreads()`.
- Optimizing memory bandwidth by reusing data from shared memory.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your stencil kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using global memory only. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `stencil_kernel` to compute the 1D stencil over the input array `a` with a given `radius`, storing the result in `c`.
3. Implement `launch_stencil` to configure the grid and block dimensions, dynamically allocate shared memory to hold the block's data plus the halo regions on both ends, and launch your kernel.
4. Make sure `launch_stencil` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
