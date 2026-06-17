# Exercise 28: Conjugate Gradient Solver

## Problem Description
**The Problem:** The Conjugate Gradient (CG) method is an iterative algorithm used to solve large sparse systems of linear equations $Ax = b$, where $A$ is a symmetric positive-definite matrix. Instead of taking steps in the direction of the local gradient like gradient descent, it takes steps in $A$-orthogonal ("conjugate") directions, guaranteeing convergence in exactly $N$ steps for an $N \times N$ matrix (though in practice it converges to a useful tolerance much faster).

**Sample Input/Output:** 
- Input: A 2x2 symmetric positive-definite matrix $A = [[4, 1], [1, 3]]$, a RHS vector $b = [1, 2]$, and an initial guess $x_0 = [0, 0]$.
- Operation: Execute the Conjugate Gradient algorithm until convergence.
- Output: The exact solution vector $x \approx [0.0909, 0.6363]$.

**Practical Importance:** 
Iterative Krylov subspace solvers like CG are the absolute foundation of large-scale computational mechanics, structural analysis (like simulating bridge stresses), and fluid dynamics. Because the matrix $A$ is huge and sparse, directly inverting it ($O(N^3)$) is impossible. CG relies heavily on Sparse Matrix-Vector Multiplication (SpMV) and vector updates. Optimizing CG means chaining these memory-bound operations together efficiently.

**Historical Anecdotes:** 
Developed by Hestenes and Stiefel in 1952, CG was originally seen as a direct solver that was too prone to rounding errors. It wasn't until the 1970s that it was recognized as a brilliant iterative solver. Today, variants of CG (like preconditioned CG or GMRES) form the core of massively parallel HPC packages like PETSc and Trilinos running on the world's fastest supercomputers.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Pointer-Based Device Variables:** To avoid devastatingly slow CPU-GPU synchronization over the PCIe bus during each iteration, we must keep scalar values (like $\alpha$ and $\beta$) on the device. Notice how `update_x_r_kernel` accepts `const float* r_dot_r` and `const float* p_Ap`. The threads compute $\alpha = (*r\_dot\_r) / (*p\_Ap)$ dynamically directly on the GPU.
- **Kernel Fusion:** While we launch several small kernels per iteration, real-world libraries often fuse these vector updates into fewer kernels to minimize global memory round trips.
- **Data Types:** *Note: HPC applications strictly use `double` precision for solvers to prevent catastrophic cancellation. We use `float` in this exercise to maintain consistency with the rest of the repository.*

## Objective
Implement a GPU-accelerated Conjugate Gradient solver.
You will learn about:
- Managing multi-kernel iterative algorithms entirely on the GPU.
- Using device-side pointers to pass intermediate scalar values between kernels without host synchronization.
- Chaining SpMV, reductions, and vector updates.

## References
- *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* by Jonathan Richard Shewchuk (A legendary, highly readable paper!).
- [CUDA Toolkit Documentation: cuSPARSE Library](https://docs.nvidia.com/cuda/cusparse/index.html)

## Files Description
- **main.cc**: The test bench. Generates a sparse Laplacian matrix, runs a CPU CG solver, and checks your kernel. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the CG steps. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Review the provided kernels: `spmv_csr_kernel`, `dot_kernel`, `update_x_r_kernel`, and `update_p_kernel`.
3. In `launch_cg`, implement the main iteration loop. 
4. Carefully map the CG mathematical steps to the corresponding kernel launches.
5. Use pointer swapping to avoid redundant `cudaMemcpy` operations for scalar values like `r_dot_r`.

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
