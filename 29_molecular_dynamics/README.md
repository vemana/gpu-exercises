# Exercise 29: Molecular Dynamics (Short-Range Forces)

## Problem Description
**The Problem:** In an N-Body simulation (Exercise 14), every particle interacts with every other particle, resulting in an $O(N^2)$ algorithm. However, in physical chemistry simulations like simulating a box of water molecules, molecules only exert significant forces on their immediate neighbors (using the Lennard-Jones potential). By defining a "cutoff radius" $r_c$, we can ignore pairs further apart, turning the algorithm from $O(N^2)$ to $O(N)$.

**Sample Input/Output:** 
- Input: Two particles in a 2D domain at positions $p_0 = (0.0, 0.0)$ and $p_1 = (1.2, 0.0)$, with cutoff radius $r_c = 2.5$.
- Operation: Compute the Lennard-Jones force vector on particle 0 from its neighbors within $r_c$.
- Output: The force on particle 0 is $F_0 \approx (2.21, 0.0)$, experiencing an attractive pull towards particle 1.

**Practical Importance:** 
Molecular Dynamics (MD) is a foundational tool in modern biochemistry and material science, used to simulate everything from protein folding to discovering new battery materials. Packages like LAMMPS and GROMACS run on the largest GPU clusters in the world. Learning how to construct dynamic neighbor lists and spatial hashes is the key to scaling these algorithms to millions of atoms.

**Historical Anecdotes:** 
The Lennard-Jones potential was proposed in 1924 by John Lennard-Jones. When scientists first started simulating it on computers in the 1950s, they could only handle a few dozen atoms. Today, state-of-the-art GPU implementations using spatial hashing can simulate billions of atoms. Spatial hashing fundamentally changed the game for particle simulations.

> [!TIP]
> **CPU Baseline:** A reference CPU implementation is available in the [`cpu_baseline`](main.cc) method of the `main.cc` file. Use this to understand the underlying logic before parallelizing it!

## Newbie Guidance
**Typical CUDA Techniques:** 
- **Spatial Hashing / Cell-Linked Lists:** To avoid an $O(N^2)$ loop, we divide the simulation domain into a grid of cells of size $r_c \times r_c$. A particle only needs to check for neighbors in its own cell and the 8 adjacent cells.
- **Atomics for Linked Lists:** To assign particles to cells on the GPU without an expensive sorting step, we can use an array `cell_starts` (storing the index of the first particle in that cell) and `particle_next` (storing the index of the next particle). By using `atomicExch` on `cell_starts`, threads can build these linked lists in parallel in a single pass!
- **Data Types:** *Note: MD applications almost exclusively use `double` precision to prevent integration errors from exploding the system over millions of timesteps. We use `float` in this exercise to maintain consistency with the rest of the repository.*

**Modern CUDA Primitives (Ampere & Beyond):**
- **Warp-Level Primitives:** While building linked lists with atomics is elegant, it suffers from poor memory coalescing during the force computation phase since linked list traversal hops randomly through memory. High-performance MD codes use `cub::DeviceRadixSort` to physically sort the particles in memory by their cell index, ensuring that threads in a warp access contiguous memory when computing forces.

## Objective
Implement a 2D Molecular Dynamics force calculation using spatial hashing.
You will learn about:
- Building cell-linked lists concurrently using atomic operations.
- Navigating linked lists stored in device memory.
- Dramatically reducing algorithmic complexity in particle simulations.

## References
- *Computer Simulation of Liquids* by Allen and Tildesley (The bible of MD algorithms)
- [LAMMPS Molecular Dynamics Simulator](https://www.lammps.org/)

## Files Description
- **main.cc**: The test bench. Scatters particles randomly in a 2D domain, computes the brute-force $O(N^2)$ forces on the CPU, and compares them against your $O(N)$ GPU spatial hashing implementation. *You should not modify this file.*
- **kernel.cu / kernel.h**: The skeleton files where you will implement the cell-building and force-calculation kernels. *This is where you will do your work.*
- **reference_kernel.cu / reference_kernel.h**: A complete, optimized reference implementation.

## What You Should Do
1. Open `kernel.cu`.
2. Implement `build_cells_kernel`. Calculate the 2D grid cell index for each particle and use `atomicExch` to prepend the particle to its cell's linked list.
3. Implement `compute_forces_kernel`. For each particle, calculate its grid cell, and loop over the 3x3 neighboring cells. Traverse the linked list of each neighboring cell to compute the Lennard-Jones forces.
4. Implement `launch_md` to allocate the temporary linked list arrays, initialize `cell_starts` to `-1`, launch the kernels, and free the temporary arrays.

## Typical Commands
- **Help Menus**:
  ```bash
  make && ./bin/run_test.sh -h
  ```
- **Test a Specific Size**:
  ```bash
  make && ./bin/run_test.sh --size 5000
  ```
- **Profile Memory Metrics**:
  ```bash
  make && ./bin/run_profiler.sh --ncu_argset=memory --size 50000
  ```
