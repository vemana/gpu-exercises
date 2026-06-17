# Exercise 36: Split-K Flash Decoding

## Problem Description
**The Problem:** Implement a Split-K Flash Attention algorithm for the decode phase ($Q$ length = 1).
You must write two kernels:
1. `split_k_pass1_kernel`: Each block processes a specific chunk of the KV cache of size `split_size`. It computes a partial row maximum $m_i$, a partial sum of exponentials $l_i$, and an unnormalized partial output vector $O_{partial}$.
2. `split_k_pass2_kernel`: Reads the partial results from all blocks, computes the global maximum $m$, global sum $l$, and outputs the final normalized attention vector $O$.

**Sample Input/Output:** 
For a 1-dimensional head ($D=1$) and sequence length $S=4$, split into 2 chunks of size 2.

**Inputs:**
Q = [1.0]
K = [0.1, 0.5, 0.9, 1.2]
V = [10.0, 20.0, 30.0, 40.0]

**Pass 1 Output (Chunk 0):**
Scores: [0.1, 0.5]
Max $m_0$: 0.5
Sum $l_0$: $\exp(0.1-0.5) + \exp(0.5-0.5) = 0.6703 + 1 = 1.6703$
Unnormalized $O_0$: $(0.6703 \times 10) + (1 \times 20) = 26.703$

**Pass 1 Output (Chunk 1):**
Scores: [0.9, 1.2]
Max $m_1$: 1.2
Sum $l_1$: $\exp(0.9-1.2) + \exp(1.2-1.2) = 0.7408 + 1 = 1.7408$
Unnormalized $O_1$: $(0.7408 \times 30) + (1 \times 40) = 62.224$

**Pass 2 (Final):**
Global Max: 1.2
Global Sum: $l_0 \times \exp(m_0 - m) + l_1 \times \exp(m_1 - m) = 1.6703 \times 0.4965 + 1.7408 \times 1 = 0.829 + 1.7408 = 2.5698$
Final $O$: $(26.703 \times 0.4965 + 62.224 \times 1) / 2.5698 = 75.48 / 2.5698 = 29.37$

**Practical Importance:** Flash Decoding is arguably the single most important optimization for LLM generation speed on long contexts. Without it, generation slows down linearly as context grows, bottlenecking purely on GPU occupancy. With Split-K Flash Decoding, generation time remains nearly constant up to extremely long contexts.

**Historical Anecdotes:** When Claude 2 and GPT-4 Turbo started supporting 100K+ token contexts, engineers quickly realized that while digesting the prompt was fast (thanks to Flash Attention), generating the response was unbearably slow. Tri Dao (the creator of Flash Attention) and researchers at PyTorch collaborated to release Flash Decoding, which instantly delivered up to 8x speedups for long-context generation, rescuing the long-context LLM revolution.

**References:** 
1. *Flash-Decoding for long-context inference* by Tri Dao et al. (2023) - The primary blog post introducing this technique.
2. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* - Foundations of sequence splitting in attention.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_attention`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
When an LLM generates text (the "decode" phase), it processes one token at a time. This means the query matrix $Q$ has a length of exactly 1! However, the context history (the KV cache) has a length $S$ which can be thousands or millions of tokens. 
If we use standard Flash Attention, we launch one thread block per query token. Since there is only 1 query token, we only launch 1 thread block! A modern GPU has over 100 Streaming Multiprocessors (SMs), meaning 99% of the GPU sits idle while one block iterates over the massive KV cache. 
**Split-K** fixes this. We split the long sequence dimension $S$ into chunks. We assign one thread block to each chunk, allowing us to compute partial attention scores in parallel across the whole GPU. A second pass then combines these partial results safely using mathematical properties of LogSumExp.

## Objective
Implement a kernel for Split-K Flash Decoding.
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
2. Implement the `split_k_flash_decoding_kernel`.
3. Implement `launch_split_k_flash_decoding` to configure the grid and block dimensions, and launch your kernel.
4. Make sure `launch_split_k_flash_decoding` returns a `std::vector<LaunchConfig>` containing the configurations for all kernels launched so the test framework can automatically display your kernel's hardware utilization.
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
