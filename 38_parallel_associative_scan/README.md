# Exercise 38: Parallel Associative Scan (Mamba / SSM)

## Problem Description
**The Problem:** Compute the recurrence: 
$$h_t = A_t h_{t-1} + B_t x_t$$
where $h_{-1} = 0$.

Define a state vector $S_t = (A_t, C_t)$, where $C_t = B_t x_t$.
Define a custom binary operator $\otimes$ such that:
$$ (A_1, C_1) \otimes (A_2, C_2) = (A_2 A_1, A_2 C_1 + C_2) $$
If you compute the inclusive prefix scan of $S$ using this operator, the second element of the result at step $t$ is exactly $h_t$!

You will implement this parallel scan using shared memory. For simplicity, we assume $N \leq 1024$ so the entire sequence fits in a single thread block.

**Sample Input/Output:** 
Let $N=3$.
Inputs:
$A = [2.0, 3.0, 0.5]$
$B = [1.0, 2.0, 1.0]$
$x = [1.0, 0.5, 4.0]$

Initial $C$:
$C_0 = B_0 \times x_0 = 1.0 \times 1.0 = 1.0$
$C_1 = B_1 \times x_1 = 2.0 \times 0.5 = 1.0$
$C_2 = B_2 \times x_2 = 1.0 \times 4.0 = 4.0$

Step-by-step sequential logic:
$h_{-1} = 0$
$h_0 = A_0 h_{-1} + C_0 = 2.0(0) + 1.0 = 1.0$
$h_1 = A_1 h_0 + C_1 = 3.0(1.0) + 1.0 = 4.0$
$h_2 = A_2 h_1 + C_2 = 0.5(4.0) + 4.0 = 6.0$

Parallel Scan Logic:
$S_0 = (2.0, 1.0)$
$S_1 = (3.0, 1.0)$
$S_2 = (0.5, 4.0)$

$S_1 \otimes S_0 = (3.0 \times 2.0, 3.0 \times 1.0 + 1.0) = (6.0, 4.0)$ -> Notice $C$ is $h_1$!
$S_2 \otimes (S_1 \otimes S_0) = (0.5 \times 6.0, 0.5 \times 4.0 + 4.0) = (3.0, 6.0)$ -> Notice $C$ is $h_2$!

**Output:**
$h = [1.0, 4.0, 6.0]$

**Practical Importance:** The parallel associative scan is the beating heart of Mamba. Without it, Mamba would be far too slow to compete with Transformers. By parallelizing the sequence dimension, Mamba achieves training speeds comparable to Flash Attention but with sub-quadratic scaling, allowing it to easily handle sequences of 1 million tokens and beyond.

**Historical Anecdotes:** For years, the AI community largely abandoned RNNs because their sequential nature meant they couldn't exploit GPU parallelism like Transformers could. Researchers Albert Gu and Tri Dao (the author of Flash Attention) realized that by enforcing certain linear constraints on the state transitions, the recurrence became mathematically associative. This small constraint allowed them to deploy parallel scan algorithms from the 1990s (like Blelloch and Kogge-Stone), single-handedly resurrecting the RNN paradigm and turning it into a state-of-the-art architecture.

**References:** 
1. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* by Albert Gu and Tri Dao (2023) - The groundbreaking paper.
2. *Parallel Prefix Sum (Scan) with CUDA* - Foundational parallel algorithms literature.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_scan`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
State Space Models (SSMs) like Mamba are the newest challengers to the Transformer architecture. Mamba processes sequences inherently like a Recurrent Neural Network (RNN). A traditional RNN seems impossible to run fast on a GPU because computing step `t` requires waiting for step `t-1` to finish. It's a sequential bottleneck!
However, with a clever mathematical trick, we can parallelize this recurrence. If we bundle the operations into a specialized "associative operator", we can use a Parallel Prefix Scan (like the Kogge-Stone algorithm). This transforms an $O(N)$ sequential loop into an $O(\log N)$ parallel tree reduction, unlocking massive GPU acceleration for Mamba!

## Objective
Implement a kernel for Parallel Associative Scan (Mamba / SSM).
You will learn about:
- Advanced kernel design and warp-level primitives.
- Hardware constraints of the GPU.

## Files Description
- **main.cc**: The test bench. It verifies your kernel's correctness against a CPU baseline, measures performance (bandwidth and time), and compares it against a reference CUDA implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement your kernel and the host-side launch logic. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation of the kernel. Used by `main.cc` to establish a performance baseline.
- **../utils/**: Contains the shared test framework, argument parsing, and utility functions.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `parallel_associative_scan_kernel`.
3. Implement `launch_parallel_associative_scan` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_parallel_associative_scan` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
