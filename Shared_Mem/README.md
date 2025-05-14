# Shared Memory and Benchmarking

Covers tiling, shared memory, and benchmarking.

## Part 1: Tiled Matrix Multiplication

 Tiled matrix multiplication is a technique that reduces the number of global memory accesses a kernel makes by storing a tile of the input matrices in shared memory.

### Implementation

Where most of the code implemenation is located:`tiled_matmul.cu`

### Testing

As with the previous lab, you can compile the code with the provided `CMakeLists.txt` by running `cmake --build build` in the terminal. Be sure to generate the test data first. If previous command does not work, simply run `bash run_tests.sh` to generate the test data and run the tests.

## Part 2: Adding Thread Coarsening

Thread coarsening can improve performance for memory-bound kernels by having each thread perform more work. This can be useful for kernels that have a high arithmetic intensity(Multiple Math computation), but are limited by memory bandwidth. Matrix multiplication is a kernel that should have high arithmetic intensity.

 That kernel should see slightly improved performance over tiled matrix multiplication due to reduced overhead from launching threads /and/ improved memory access patterns.

### Implementation

 A separate folder is available for this part. Implement your kernel in `coarse_matmul.cu`. The implementation is similar to the tiled matrix multiplication kernel.

### Testing

Compile the code with the provided `CMakeLists.txt` by running `cmake --build build` in the terminal. Be sure to generate the test data first. If your experience any issue, simply run `make run_tests.sh` to generate the test data and run the tests. Once you've passed all the tests, your kernel is ready for benchmarking.

## Part 3: Benchmarking

Once all matrix multiplication kernels have been added and tested, you can benchmark the performance of each kernel using NSight Compute. A benchmarking script is provided for you in `benchmark.sh`. You can run the script by executing `bash benchmark.sh` in the terminal. This will generate a `.ncu-rep` file that you can open in NSight Compute to analyze the performance of your kernels. It may take a while to run as it will test each kernel using a large input size.

Once the run is complete, load the `.ncu-rep` file into NSight Compute and analyze the performance of each kernel.
