# Exercise 27: Fast Fourier Transform (1D Radix-2)

## Problem Description
**The Problem:** The Fast Fourier Transform (FFT) efficiently computes the Discrete Fourier Transform (DFT) of a sequence. It reduces the computational complexity from $O(N^2)$ to $O(N \log N)$. Here, we will implement the Cooley-Tukey Radix-2 Decimation-In-Time (DIT) algorithm for 1D sequences of size $N=2^m$.

**Sample Input/Output:** 
- Input: A sequence of 4 complex numbers (time domain): `[(1, 0), (1, 0), (-1, 0), (-1, 0)]`
- Operation: Perform a 1D Radix-2 FFT.
- Output: The sequence in the frequency domain: `[(0, 0), (2, -2), (0, 0), (2, 2)]`

**Practical Importance:** 
The FFT is often called the most important numerical algorithm of the 20th century. It is the heart of digital signal processing (audio, radio, radar), image compression (JPEG), and solving partial differential equations (spectral methods). In HPC, massive 3D FFTs are required for Molecular Dynamics simulations (Particle-Mesh Ewald) and Density Functional Theory (quantum mechanics). High-performance FFT libraries like NVIDIA's cuFFT heavily leverage shared memory and specialized warp instructions to maximize memory bandwidth utilization.

**Historical Anecdotes:** 
While Cooley and Tukey popularized the algorithm in 1965, Carl Friedrich Gauss actually discovered it independently around 1805 to calculate the orbits of asteroids! Today, executing FFTs on GPUs can be hundreds of times faster than CPU equivalents because the algorithm's recursive structure maps elegantly to parallel threads and memory hierarchies.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Bit-Reversal Permutation:** The Radix-2 DIT FFT requires the input data to be reordered. The index $i$ must be swapped with the index formed by reversing the binary bits of $i$. CUDA provides intrinsic functions like `__brev()` to reverse bits extremely fast.
- **The "Butterfly" Pattern:** The core operation of the FFT is the "butterfly," which takes two complex numbers, multiplies one by a "twiddle factor" (a complex root of unity), and computes their sum and difference.
- **Data Types:** *Note: HPC applications typically use `double` precision (cuDoubleComplex) for FFTs. We use `float` (cuFloatComplex) in this exercise to maintain consistency with the rest of the repository.*

**Modern CUDA Primitives (Ampere & Beyond):**
- **Shared Memory:** The naive implementation provided here performs $O(\log N)$ global memory reads/writes per element (one per stage). Modern implementations load small blocks of the array into Shared Memory, perform several stages of the butterfly entirely within the SM, and then write the result back to global memory. This drastically reduces the memory bandwidth bottleneck.

## Objective
Implement a 1D Radix-2 Cooley-Tukey FFT.
You will learn about:
- Managing complex numbers in CUDA (`cuFloatComplex`).
- Utilizing bit-manipulation intrinsics for permutation arrays.
- Coordinating threads to execute a multi-stage dependency graph (the butterfly network).

## References
- *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu (Chapter 18: Parallel Fast Fourier Transform)
- [NVIDIA cuFFT Library Documentation](https://docs.nvidia.com/cuda/cufft/index.html)

## Files Description
- **main.cc**: The test bench. Generates a mixed sine wave signal, executes a CPU DFT for correctness validation, and measures your kernel's performance. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the bit-reversal and butterfly kernels. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Implement `bit_reverse_permutation_kernel` using `__brev()` to reorder the `x` array in-place.
3. Implement `fft_stage_kernel` to compute a single stage of the Radix-2 butterfly network. Ensure you correctly calculate the twiddle factor $W_m^k$.
4. Implement `launch_fft`. It should launch the bit-reversal kernel once, and then loop to launch the `fft_stage_kernel` exactly $\log_2 N$ times.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 4096
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 4096
  ```
