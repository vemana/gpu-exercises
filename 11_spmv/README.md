# Exercise 11: Sparse Matrix-Vector Multiplication (SpMV)

## Problem Description
**The Problem:** Sparse Matrix-Vector Multiplication (SpMV) computes $y = A x$, where $A$ is a large matrix with mostly zero elements. Storing and computing the zeros is wasteful, so $A$ is compressed (e.g., Compressed Sparse Row - CSR format). Parallelizing this requires threads to navigate the compressed data structures and handle rows with vastly different numbers of non-zero elements.

**Sample Input/Output:**
- Dense Matrix $A$: `[[1, 0, 2], [0, 0, 3], [4, 5, 6]]`
- Vector $x$: `[1, 2, 3]`
- Output $y$: `[7, 9, 32]`

**Practical Importance:** SpMV is the most time-consuming operation in iterative solvers for massive systems of linear equations. It is heavily used in finite element methods (structural engineering, fluid dynamics), economic modeling, and machine learning (e.g., Graph Neural Networks, where the adjacency matrix is highly sparse).

**Historical Anecdotes:** The study of sparse matrices took off in the 1960s and 70s as engineers tried to simulate larger physical systems than their limited computer memory could hold. When GPUs entered the scene, SpMV became a premier battleground for optimization. The irregular memory access patterns of SpMV perfectly highlight the tension between GPU compute power and memory bandwidth limitations.

**References:**
- *Programming Massively Parallel Processors* (Chapter: Sparse Matrix Computation)
- *Efficient Sparse Matrix-Vector Multiplication on CUDA* by Nathan Bell and Michael Garland

## Newbie Guidance
**Typical CUDA Techniques:**
- **Load Balancing (Vector vs. Scalar CSR):** A naive approach assigns one thread per row (Scalar CSR). If one row has 1,000 non-zeros and another has 2, that warp is severely divergent. A better approach (Vector CSR) assigns a whole warp to process a single row, performing a parallel reduction to sum the row's dot product.
- **Memory Coalescing:** The vector $x$ is gathered randomly, making SpMV memory-bound. Restructuring data to ELLPACK or coordinate (COO) format can sometimes yield better coalescing than CSR.

**Modern CUDA Primitives (Ampere & Beyond):**
- **Sparse Tensor Cores:** Ampere introduced hardware support for 2:4 structured sparsity. If your matrix can be pruned so that exactly 2 out of every 4 elements are non-zero, you can use specialized `mma.sp` instructions to execute the SpMV on Tensor Cores at double the throughput.
- **Warp Reduce:** The `__reduce_add_sync()` warp intrinsic makes the row-summation phase of Vector CSR virtually instant.

## Objective
Implement Sparse Matrix-Vector Multiplication (SpMV) using the Compressed Sparse Row (CSR) format.
You will learn about:
- Irregular memory access patterns.
- Load balancing threads when data structure sparsity varies.
- Dealing with uncoalesced memory accesses inherent to sparse graphs.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your SpMV kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized naive reference implementation where one thread computes exactly one row. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions for checking correctness and computing kernel occupancy/utilization metrics.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `spmv_kernel` to compute `y = A * x` where `A` is represented by `values`, `col_indices`, and `row_offsets`.
3. Try optimizing the naive approach! Can you assign a warp to each row instead of a single thread? This is called a "Warp-centric" or "Vector" SpMV algorithm.
4. Implement `launch_spmv` to configure the grid and block dimensions, calculate occupancy metrics, and launch your kernel.
5. Make sure `launch_spmv` returns the populated `LaunchMetrics` struct so the test framework can automatically display your kernel's hardware utilization.
6. Compile using `make` and run `./bin/run_test.sh` to evaluate your correctness and performance, and `./bin/run_profiler.sh` to identify bottlenecks.

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
