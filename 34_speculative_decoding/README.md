# Exercise 34: Speculative Decoding (Parallel Rejection Sampling)

## Problem Description
**The Problem:** Large Language Models (LLMs) are severely memory-bandwidth bound during token generation because the entire model weights (often >100GB) must be loaded from HBM to compute just a single output token. **Speculative Decoding** solves this by running a tiny "draft" model to quickly guess $N$ tokens. The large target model then evaluates all $N$ tokens in parallel. If the draft was good, the target model accepts them, generating $N$ tokens in the time it usually takes to generate 1!

**Sample Input/Output:** 
- Input: We have $N=4$ draft tokens. The target model outputs probabilities `p=[0.9, 0.8, 0.1, 0.9]` for those specific tokens. The draft model's probabilities were `q=[0.6, 0.8, 0.8, 0.9]`. The random variables are `r=[0.5, 0.1, 0.7, 0.5]`.
- Operation: For each token, if $r < p/q$, accept it. 
  - Token 0: $0.5 < 0.9/0.6=1.5$ (Accept)
  - Token 1: $0.1 < 0.8/0.8=1.0$ (Accept)
  - Token 2: $0.7 < 0.1/0.8=0.125$ (REJECT)
  - Token 3: Skipped because Token 2 was rejected.
- Output: `accept_length = 2`.

**Practical Importance:** 
Speculative decoding can yield a 2-3x speedup in LLM serving without degrading the mathematical output distribution. It is widely used by frontier AI labs (DeepMind originally published it for Chinchilla) and commercial serving infrastructure (vLLM, TGI). The parallel verification step we implement here is critical to realizing those speedups.

**Historical Anecdotes:** 
The core mathematical trick of speculative decoding relies on a classic statistics technique called "Rejection Sampling", invented by John von Neumann in 1951. Applying it to autoregressive Transformers in 2023 was a massive breakthrough because it circumvented the seemingly unbreakable serial dependency of text generation (where token $N$ strictly depends on token $N-1$).

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Warp-Level Voting & Bit Manipulation:** Notice the dependency: if token 2 is rejected, tokens 3 and 4 MUST be rejected even if their math works out. Doing this sequentially is slow. Instead, each thread in a warp can independently evaluate its token. Then, using `__ballot_sync`, the warp creates a 32-bit integer where a `1` means "rejected". Finally, the hardware intrinsic `__ffs` (Find First Set) instantly locates the lowest rejected index! 
- **Warp Execution:** Because a standard draft length is 8 tokens, a single warp (32 threads) can evaluate an entire batch. By assigning 1 warp per batch, we avoid grid-level synchronizations entirely.

## Objective
Implement parallel rejection sampling for Speculative Decoding.
You will learn about:
- Prefix dependencies and breaking sequential bottlenecks.
- Warp-level communication primitives (`__ballot_sync`).
- Bit manipulation intrinsics (`__ffs`).

## References
- *Accelerating Large Language Model Decoding with Speculative Sampling* by Chen et al. (DeepMind)
- *Fast Inference from Transformers via Speculative Decoding* by Leviathan et al. (Google Research)

## Files Description
- **main.cc**: The test bench. Generates target probabilities, draft probabilities, and random variables. Validates your parallel acceptance length against a sequential CPU baseline. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the warp voting algorithm. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Notice that the kernel maps 1 warp (32 threads) to 1 batch.
3. Compute the `reject` boolean for your thread's assigned token. Ensure inactive lanes (beyond `draft_len`) do not interfere.
4. Use `__ballot_sync` with the active mask to generate a 32-bit `reject_mask`.
5. Thread 0 (lane 0) should use `__ffs(reject_mask) - 1` to find the index of the first rejection and write it to `accept_lengths`.
6. If the mask is `0` (all accepted), write `draft_len`.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 1000
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 1048576
  ```
