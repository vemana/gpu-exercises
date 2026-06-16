# Exercise 03: Exclusive Prefix Sum (Scan)

## Problem Description
**The Problem:** Prefix Sum (or Scan) takes an array and computes a new array where each element at index $i$ is the sum of all elements before it (exclusive scan) or up to it (inclusive scan). While it looks like an inherently sequential $O(N)$ loop, it can be parallelized into a tree-based $O(N \log N)$ algorithm, or a work-efficient $O(N)$ algorithm.

**Sample Input/Output (Exclusive Sum):**
- Input `A`: `[3, 1, 7, 0, 4, 1, 6, 3]`
- Output `C`: `[0, 3, 4, 11, 11, 15, 16, 22]`

**Practical Importance:** Scan is the "Swiss Army Knife" of parallel algorithms. It is fundamentally used to turn dynamic, variable-length outputs into a dense, compacted array. It enables stream compaction (filtering data), radix sort, polynomial evaluation, and solving recurrences. In graphics, it is used for rendering dynamic particle systems; in databases, it enables blazing-fast query filtering and indexing.

**Historical Anecdotes:** The parallel prefix sum algorithm was first developed in the 1960s for digital circuit design (carry-lookahead adders). Later, Guy Blelloch heavily popularized it in 1990 for data-parallel computing, demonstrating that many seemingly sequential problems could be solved efficiently in parallel using Scan.

**References:**
- Guy E. Blelloch's 1990 paper: [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- GPU Gems 3, Chapter 39: *Parallel Prefix Sum (Scan) with CUDA* by Mark Harris

## Newbie Guidance
**Typical CUDA Techniques:**
- **Double Buffering:** In shared memory, use double buffering (ping-ponging between two arrays) to prevent read-after-write hazards during the iterative steps of a scan without needing massive numbers of barrier synchronizations.
- **Bank Conflicts:** Shared memory is divided into banks. If multiple threads access different addresses in the same bank, accesses are serialized. Padding arrays to offset index calculations is a classic trick to prevent this.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp Shuffle Instructions:** Similar to reduction, intra-warp scanning is best done via `__shfl_up_sync()`, allowing rapid bitwise prefix summing without touching shared memory.
- **Grid Synchronization / TMA:** To compute a global scan of an array larger than one block, you typically need multiple kernel launches (scan blocks -> scan block sums -> add sums to blocks). On Hopper, Thread Block Clusters and Distributed Shared Memory (DSMEM) enable different blocks to communicate their partial sums directly, allowing a global scan to be done in a single kernel launch.

## Objective
Implement an exclusive prefix sum (scan) algorithm.
You will learn about:
- Work-efficient parallel algorithms (Blelloch scan).
- Avoiding bank conflicts in shared memory.
- Multi-block synchronization (if implementing global scan).

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your scan kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation (using Thrust). Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `scan_kernel` to compute the exclusive prefix sum of array `a` into array `c`.
3. Implement `launch_scan` to configure the grid and block dimensions, dynamically allocate shared memory (if needed), calculate occupancy metrics, and launch your kernel.
4. Make sure `launch_scan` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
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
