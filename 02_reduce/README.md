# Exercise 02: Parallel Reduction

## Problem Description
**The Problem:** The Reduce pattern combines all elements in a dataset into a single value using an associative operator (e.g., sum, max, min). It transforms $N$ elements into $1$ element. This requires careful synchronization because threads must coordinate their partial results.

**Sample Input/Output:**
- Input `A`: `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]`
- Operator: `+`
- Output: `36.0`

**Practical Importance:** Reduction is ubiquitous in scientific computing and machine learning. Calculating the total energy in a physics simulation, finding the maximum error in a solver, or computing the gradient norm in neural network backpropagation all rely on reduction. In modern AI, calculating the softmax denominator over a vocabulary of 100,000 tokens requires extremely fast reductions to prevent the GPU from stalling.

**Historical Anecdotes:** Reduction operations were a key component of APL (A Programming Language) in the 1960s, denoted by the slash operator (`/`). When GPUs were first being hacked to do non-graphics work in the early 2000s, writing an efficient reduction was considered a rite of passage because it was notoriously difficult to do without atomic operations (which were extremely slow on early hardware).

**References:**
- *Optimizing Parallel Reduction in CUDA* by Mark Harris (NVIDIA): A classic, definitive guide on tree-based reductions.
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Reduction)

## Newbie Guidance
**Typical CUDA Techniques:**
- **Shared Memory:** A naive reduction constantly writes to global memory. A fast reduction pulls a tile of data into shared memory, reduces it locally within a thread block, and only writes the block's single partial sum back to global memory.
- **Warp-Level Primitives:** Within a warp (32 threads), shared memory isn't even strictly necessary. You can use warp shuffle instructions (`__shfl_down_sync()`) to drastically speed up the final stages of the reduction tree without hitting memory at all.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Cooperative Groups:** The `cooperative_groups::reduce` primitive provides a safe, highly optimized, and hardware-accelerated way to reduce data across an entire thread block or even an entire grid, avoiding the need to write custom tree-reduction logic.
- **Asynchronous Barriers:** Ampere introduced `cuda::barrier`, allowing finer-grained synchronization than `__syncthreads()`. You can compute while waiting for memory to arrive. On Hopper, Distributed Shared Memory (DSMEM) allows threads to access the shared memory of other blocks in the same cluster, which can be leveraged for faster multi-block reductions.

## Objective
Implement a parallel reduction algorithm (sum reduction) using CUDA shared memory. 
You will learn about:
- Tree-based reduction to avoid warp divergence.
- Utilizing shared memory for fast intra-block communication.
- Synchronizing threads within a block.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your parallel reduction kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation of the reduction kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `reduce_kernel` using shared memory.
3. Implement `launch_reduce` to configure the grid and block dimensions, dynamically allocate shared memory (if needed), and launch your kernel.
4. Make sure `launch_reduce` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
