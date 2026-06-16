# Exercise 08: Stream Compaction

## Problem Description
**The Problem:** Stream Compaction (or filtering) takes an input array and a boolean condition, and produces a new array containing only the elements that satisfy the condition, preserving their original relative order. In a parallel setting, this is non-trivial because threads don't know *where* to write their accepted elements without coordinating with others.

**Sample Input/Output:**
- Input `A`: `[3, 0, 7, 0, 4, 0, 6, 3]`
- Condition: `x != 0`
- Output `C`: `[3, 7, 4, 6, 3]`

**Practical Importance:** Compaction is critical for keeping GPUs fed with useful work. When simulating particles, you want to remove dead particles from the array so subsequent kernels don't waste threads processing them. In graphics, it is used in deferred shading to cull pixels that don't need lighting. In database queries, it is the fundamental `SELECT ... WHERE ...` operation.

**Historical Anecdotes:** Stream compaction heavily relies on the Exclusive Prefix Sum (Scan) operation. Once Guy Blelloch demonstrated how to efficiently scan in parallel in 1990, stream compaction became a classic building block. It transformed the way algorithms were written, encouraging developers to aggressively filter datasets between compute passes to maximize arithmetic density.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Parallel Patterns: Stream Compaction)
- GPU Gems 3, Chapter 39: *Parallel Prefix Sum (Scan) with CUDA*

## Newbie Guidance
**Typical CUDA Techniques:**
- **Scan-then-Scatter:** The standard implementation involves: 1) Evaluate the condition to a boolean array. 2) Perform an Exclusive Prefix Sum (Scan) on the boolean array to compute the exact destination index for every valid element. 3) Scatter the valid elements to their new indices.
- **Warp-Level Aggregation:** To reduce global memory traffic, threads can first count how many elements their warp accepts using `__ballot_sync()`, compute a single warp-level offset, and write compactly.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp Match & Reduce:** The `__match_any_sync` and `cooperative_groups` APIs allow warps to quickly figure out who holds valid data without heavy arithmetic.
- **Hardware Asynchronous Barriers:** On Hopper, managing the multi-phase nature of Compaction (evaluate -> sync -> scan -> sync -> scatter) is vastly accelerated using `cuda::barrier` to overlap the phases across Thread Block Clusters.

## Objective
Implement a stream compaction algorithm to filter an array.
You will learn about:
- Combining prefix sum (scan) with scatter operations.
- Building a work-efficient filter using multiple phases.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your compaction kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using Thrust. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `compaction_kernel` (or a combination of kernels) to filter the array `a` such that only elements `> 0` are kept, writing them contiguously to `c`, and writing the total valid count to `count`.
3. Implement `launch_compaction` to launch the necessary sequence of kernels (e.g., predicate generation, scan, scatter), calculate occupancy metrics, and return the final `LaunchMetrics`.
4. Make sure `launch_compaction` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
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
