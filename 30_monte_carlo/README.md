# Exercise 30: Monte Carlo Integration (Options Pricing)

## Problem Description
**The Problem:** Analytical solutions to complex systems are often impossible. Instead, we can use the Monte Carlo method: simulate thousands of random scenarios and take the average outcome. In this exercise, we will calculate the price of a European Call Option (Black-Scholes model) by simulating millions of random stock price paths.

**Sample Input/Output:** 
- Input: Initial stock price $S_0 = 100$, Strike $K = 100$, Risk-free rate $r = 0.05$, Volatility $\sigma = 0.2$, Time $T = 1.0$.
- Operation (Single Path): A random normal $z = 0.5$ is generated. $S_T = 100 \times \exp((0.05 - 0.5 \times 0.2^2) \times 1.0 + 0.2 \times \sqrt{1.0} \times 0.5) = 113.88$. The payoff is $\max(113.88 - 100, 0) = 13.88$.
- Output: Averaging millions of these discounted payoffs yields the final option price $\approx 10.45$.

**Practical Importance:** 
Monte Carlo simulations are heavily utilized in quantitative finance (pricing exotic derivatives), physics (particle transport, quantum chromodynamics), and computer graphics (path tracing). Because each random path is completely independent of the others, Monte Carlo is "embarrassingly parallel" and incredibly well-suited for GPUs. The main challenge is generating random numbers fast enough!

**Historical Anecdotes:** 
The method was invented by Stanislaw Ulam at Los Alamos in 1946 while playing solitaire while recovering from an illness. He realized it was easier to play 100 games and count the wins than to calculate the combinatorial probability analytically. He named it after the Monte Carlo Casino in Monaco. 

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Parallel Random Number Generation (PRNG):** Generating random numbers in parallel is hard. If two threads use the same state, they produce identical paths. We use a deterministic PCG hash where each thread uses its `tid` as a unique seed. For production use, NVIDIA provides the highly optimized `cuRAND` library.
- **The Box-Muller Transform:** Our PRNG produces uniformly distributed numbers in $(0, 1)$. We need normally distributed numbers $N(0, 1)$ to simulate random stock walks. The Box-Muller transform turns two uniforms into a normal variable.
- **Data Types:** *Note: While `double` precision is safer for massive reductions, we use `float` to stay consistent with the repository.*

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp-Level Reductions:** After generating millions of payoffs, we need to sum them up. You could use `atomicAdd` for every single thread, but that destroys performance. Instead, use Warp Shuffle instructions (`__shfl_down_sync`) to sum the values within a 32-thread warp in just 5 steps, then use shared memory to combine the warp sums into a block sum, and finally perform a single `atomicAdd` per block to global memory.

## Objective
Implement a GPU-accelerated Monte Carlo options pricer.
You will learn about:
- Generating pseudo-random numbers in parallel.
- Performing mathematical transforms (Box-Muller) in kernels.
- Combining local computations with highly efficient warp and block-level reductions.

## References
- [NVIDIA cuRAND Library Documentation](https://docs.nvidia.com/cuda/curand/index.html)
- *Paul Glasserman: Monte Carlo Methods in Financial Engineering*

## Files Description
- **main.cc**: The test bench. Computes the Monte Carlo sum sequentially on the CPU and tests your kernel's precision and speed. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the path generation and reduction. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Implement the `monte_carlo_kernel`. Use the provided `rand_uniform` function to generate two uniform variables.
3. Apply the Box-Muller transform to get $z$.
4. Calculate the final stock price $S_T$ and the payoff.
5. Use warp shuffle instructions to reduce the payoffs within the warp, then use shared memory to reduce across the block.
6. The thread with `laneId == 0` and `warpId == 0` should use `atomicAdd` to add the block's sum to `d_sum`.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 1000000
  ```
- **Profile Compute Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=compute --size 10000000
  ```
