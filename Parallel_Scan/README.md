# Parallel Scan

Covers inclusive parallel scan.

# Objective

This kernel performs an inclusive parallel scan on a 1D list. The scan operator will be the addition (plus) operator. This kernel handle input lists of arbitrary length.  This means that the computation can be performed using only one kernel launch.

The boundary condition can be handled by filling "identity value (0 for sum)" into the shared memory of the last block when the length is not a multiple of the thread block size.

# Information about Kernel

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the work efficient scan routine
- use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory


# Build Instructions

The data and tests can be generated and run in the same way as in previous labs.(e.g cmake, build build, bash command)
