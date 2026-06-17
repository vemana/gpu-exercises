# Exercise 33: Grouped-Query Attention (GQA)

## Problem Description
**The Problem:** Standard Multi-Head Attention (MHA) has identical numbers of Query (Q), Key (K), and Value (V) heads. This means the KV cache memory footprint grows enormous during LLM inference. Multi-Query Attention (MQA) solved this by having all Q heads share exactly 1 K and 1 V head, drastically reducing memory but slightly degrading model quality. **Grouped-Query Attention (GQA)** strikes the perfect balance: it groups a subset of Q heads to share 1 KV head (e.g., 32 Q heads sharing 8 KV heads).

**Sample Input/Output:** 
- Input: Q heads = 32, KV heads = 8. Group ratio = $32/8 = 4$. Query head `13` needs to fetch its K and V states.
- Operation: The kernel determines the corresponding KV head index using `kv_idx = q_idx / group_ratio`. For $q=13$, it fetches from KV head `13 / 4 = 3`.
- Output: The correctly mapped context vector utilizing the broadcasted KV states.

**Practical Importance:** 
GQA is the de-facto standard for all modern frontier models, including Llama-3, Mistral, and Gemma. By reducing the number of KV heads, the memory footprint of the KV cache drops by $8	imes$ (if ratio is 8). This allows larger batch sizes to be served on a single GPU, drastically improving memory-bound throughput and lowering cloud costs.

**Historical Anecdotes:** 
GQA was popularized by Ainslie et al. (Google Research) in 2023. They proved that GQA achieves quality nearly identical to MHA but with the blazing fast inference speed of MQA. Almost overnight, the entire open-weight AI community abandoned standard MHA in favor of GQA.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **The Online Softmax Trick (FlashAttention):** Computing attention the naive way requires allocating an array of size $seq\_len$ to store the intermediate dot products (logits) before applying Softmax. We instead use the FlashAttention mathematical trick: maintaining a running maximum ($m$) and running sum ($l$) in thread registers. This reduces the memory footprint from $O(N)$ to $O(1)$, completely removing the need for intermediate Shared/Global memory!
- **Data Types:** *Note: LLMs exclusively use `half` (FP16) or `bfloat16`. We use `float` in this exercise for simplicity.*

## Objective
Implement a Grouped-Query Attention kernel utilizing the online-softmax memory optimization trick.
You will learn about:
- Grouping logic to broadcast Key/Value states across multiple Query heads.
- Implementing the FlashAttention online Softmax trick to compute attention in a single pass without extra memory.

## References
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* by Ainslie et al.
- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* by Dao et al.

## Files Description
- **main.cc**: The test bench. Generates random Q, K, V arrays and tests your GQA implementation against a naive CPU baseline. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the group routing and online softmax computation. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Notice the grid maps `seq_len` to the X dimension and `q_idx` to the Y dimension.
3. Calculate the correct `kv_idx` using the group ratio.
4. Implement the single-pass loop over the sequence length. Calculate the dot product between $Q$ and $K$.
5. Use the provided $m$, $l$, and $out\_vec$ registers to implement the online Softmax state updates exactly as defined in the FlashAttention paper.
6. Write the final accumulated $out\_vec$ divided by the normalization factor $l$ to global memory.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 256
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 1024
  ```
