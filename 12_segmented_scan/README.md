# Exercise 12: Segmented Scan

## Problem Description
**The Problem:** Segmented Scan applies the prefix sum operation to an array, but resets the running sum at specified boundaries (segments). It allows multiple independent scans of different lengths to be computed simultaneously within the same array, maximizing GPU utilization.

**Sample Input/Output:**
- Input `A`: `[1, 2, 3, 4, 5, 6]`
- Segment Flags: `[1, 0, 0, 1, 0, 0]` (1 indicates a new segment)
- Output `C`: `[1, 3, 6, 4, 9, 15]`

**Practical Importance:** Segmented scan is a vital tool for handling nested parallelism on GPUs. For example, in graph algorithms, a single array might contain the neighbor lists for all vertices (which have different lengths). Segmented scan allows you to process all these variable-length lists in a single massive parallel operation. It's also used in QuickSort implementations, computational geometry, and parsing irregular data strings.

**Historical Anecdotes:** Segmented Scan was introduced by Guy Blelloch in his NESL language (a parallel functional programming language) to seamlessly flatten nested data structures. It is widely considered an elegant, advanced technique that separates intermediate GPU programmers from the masters, as it transforms deeply irregular loop structures into a single flat operation.

**References:**
- Guy Blelloch's thesis: *Vector Models for Data-Parallel Computing*
- *CUDPP: CUDA Data Parallel Primitives Library*

## Newbie Guidance
**Typical CUDA Techniques:**
- **Flag Logic:** A segmented scan is just a regular exclusive/inclusive scan, but the addition operator is modified: `if (flag_b == 1) sum = val_b; else sum = val_a + val_b;`. 
- **Shared Memory:** Like regular scan, this must be done in shared memory first. You propagate both the running sums and the segment flags up the reduction tree.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp Shuffle:** As with regular scan, the base level of the segmented scan should be executed in registers using `__shfl_up_sync()`, incorporating the flag logic via predicated execution.
- **Asynchronous Execution (Hopper):** Since segmented scans often act as a pre-processing step for variable-length data processing (like graph neighbor lists), utilizing Hopper's asynchronous barriers allows you to immediately begin processing a segment as soon as its length is computed, overlapping the scan with the actual compute payload.

## Objective
Implement a segmented prefix sum (scan) operation.
You will learn about:
- Modifying standard parallel algorithms to handle boundary conditions.
- Fusing operations (scan + segment flags check) to save memory bandwidth.
- Complex intra-block and inter-block synchronization.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your segmented scan kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation using Thrust. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `segmented_scan_kernel` (or a combination of kernels) to compute the inclusive prefix sum of array `a` into array `c`, resetting the sum whenever the corresponding element in `flags` is `1`.
3. Implement `launch_segmented_scan` to configure the grid and block dimensions, calculate occupancy metrics, and return the final `LaunchMetrics`.
4. Make sure `launch_segmented_scan` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
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
