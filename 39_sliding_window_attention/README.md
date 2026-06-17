# Exercise 39: Sliding Window Attention

## Problem Description
**The Problem:** Implement a Sliding Window Attention kernel.
Given a query sequence $Q$ of length $S$, and corresponding keys and values $K$ and $V$, compute the attention output $O$.
The kernel is causal, meaning token $i$ only attends to tokens $j \le i$.
However, the SWA constraint dictates that token $i$ can only attend to tokens $j \ge \max(0, i - W + 1)$, where $W$ is the window size.
Everything outside this banded region receives an attention score of $-\infty$ (or is simply skipped in the loop).

**Sample Input/Output:** 
Let $S = 4, D = 1, W = 2$.
Inputs:
$Q = [1.0, 1.0, 1.0, 1.0]$
$K = [2.0, 3.0, 4.0, 5.0]$
$V = [10.0, 20.0, 30.0, 40.0]$

**Computation for Token 2 (index 2):**
Normally, token 2 attends to {0, 1, 2}.
But window $W = 2$. Valid bounds: $\max(0, 2 - 2 + 1) = 1$. So it only attends to {1, 2}.
Scores:
$S_{2, 1} = Q_2 \times K_1 = 1.0 \times 3.0 = 3.0$
$S_{2, 2} = Q_2 \times K_2 = 1.0 \times 4.0 = 4.0$

Softmax over {3.0, 4.0}:
$P_1 = \exp(3.0) / (\exp(3.0) + \exp(4.0)) = 0.2689$
$P_2 = \exp(4.0) / (\exp(3.0) + \exp(4.0)) = 0.7311$

Output $O_2$:
$O_2 = P_1 V_1 + P_2 V_2 = 0.2689 \times 20.0 + 0.7311 \times 30.0 = 5.378 + 21.933 = 27.31$

**Expected Output for Token 2:** [27.31]

**Practical Importance:** Sliding Window Attention drastically reduces the memory bandwidth requirement for the KV Cache. For Mistral 7B, a window of 4096 means that regardless of whether the context is 10k or 100k tokens, the computational cost of attention plateaus at a constant boundary per token. This is how small models efficiently digest entire books!

**Historical Anecdotes:** When the Mistral 7B model dropped out of nowhere in late 2023, it outperformed Llama-2 13B and shocked the world. One of its "secret sauces" was the heavy use of Sliding Window Attention and Grouped Query Attention. It proved that attention didn't need to be fully global across a massive context if the model was trained cleverly. Mistral released a highly optimized CUDA kernel via `vLLM` to ensure the community could actually run their model fast.

**References:** 
1. *Mistral 7B* by Albert Jiang et al. (2023) - The architecture paper that popularized SWA for modern powerful open-weight LLMs.
2. *Longformer: The Long-Document Transformer* by Iz Beltagy et al. (2020) - One of the earliest papers detailing local banded attention patterns.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_sliding_window_attention`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
Standard self-attention is famous for its $O(N^2)$ complexity. If a text sequence gets twice as long, it requires four times as much compute and memory! To solve this, researchers designed "Sliding Window Attention". 
Instead of a token looking at *every* token that came before it, it is restricted to looking only at the last $W$ tokens (its "window"). If $W=4000$, and the sequence is $10000$ tokens long, token 10000 only attends to tokens 6000 through 10000. 
In a GPU kernel, implementing this involves cleverly adjusting the loop boundaries. It sounds trivial, but doing this without disrupting the memory access coalescing and warp scheduling is a fun challenge!

## Objective
Implement a kernel for Sliding Window Attention.
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
2. Implement the `sliding_window_attention_kernel`.
3. Implement `launch_sliding_window_attention` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_sliding_window_attention` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
