# Exercise 32: Mixture of Experts (MoE) Top-K Routing

## Problem Description
**The Problem:** Scaling up the parameters of a Large Language Model (LLM) usually causes the computational cost to explode. A Sparse Mixture of Experts (MoE) model bypasses this by having many "expert" feed-forward networks, but only activating a small subset (e.g., $K=2$) of them for each token. A small gating network outputs probabilities for which expert is best suited for each token. We must efficiently compute the top $K$ experts and their routing multipliers.

**Sample Input/Output:** 
- Input: For a single token, a gating network outputs logits for 8 experts: `[1.2, -0.5, 3.4, 0.1, 2.8, -1.1, 0.0, 0.9]`. We want $K=2$.
- Operation: Select the top 2 values (`3.4` for expert 2, and `2.8` for expert 4). Compute the softmax over just those two values: $\exp(3.4) / (\exp(3.4) + \exp(2.8))$ and $\exp(2.8) / (\exp(3.4) + \exp(2.8))$.
- Output: Top Experts: `[2, 4]`. Routing Weights: `[0.6456, 0.3543]`.

**Practical Importance:** 
MoE is arguably the most critical architecture for frontier models. Models like Mistral's Mixtral 8x7B or GPT-4 use MoE to achieve massive capacity while maintaining fast inference speeds. Because this routing step determines which memory blocks contain the expert weights needed next, doing this quickly and correctly is absolutely essential for LLM inference latency.

**Historical Anecdotes:** 
The concept of Mixture of Experts dates back to 1991 (Jacobs et al.), but it was the 2017 paper "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" by Shazeer et al. at Google that proved MoE could scale deep learning models to astronomical parameter counts. Today, sparse routing is the secret sauce behind the fastest open-weight models.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Thread-Level Sorting:** When $E$ (the number of experts) is small (e.g., 8), a single thread can loop through the array and select the top $K$ values. This avoids expensive block-level sorting or reductions.
- **Sparse Token Permutation (Context):** In a real MoE layer, after determining the routing destinations for all tokens in a batch, the tokens must be physically rearranged in memory using a radix sort (e.g., `cub::DeviceRadixSort`) so that all tokens going to Expert 0 are contiguous, all going to Expert 1 are contiguous, etc. This routing kernel is step 1 of that pipeline.

## Objective
Implement the top-K routing logic for a Mixture of Experts layer.
You will learn about:
- Thread-local reduction and selection algorithms.
- Calculating sparse softmax operations.
- The foundational mechanics of modern frontier LLM scaling.

## References
- *Mixtral of Experts* by Jiang et al.
- *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* by Fedus et al.

## Files Description
- **main.cc**: The test bench. Generates random logits for millions of tokens and validates your Top-2 selection and normalization against a CPU baseline. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the thread-local selection. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Notice that the kernel maps exactly 1 token to 1 thread.
3. For your thread's token, iterate through its $E=8$ expert logits to find the largest and second-largest values.
4. Keep track of the indices of those top 2 values.
5. Compute the softmax over those two values to produce the routing weights.
6. Write the top 2 indices and their corresponding weights to the output arrays.

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
