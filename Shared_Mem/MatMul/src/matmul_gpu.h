#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// sgemm stands for "single-precision general matrix multiply"
// This follows the naming convention of the BLAS library
int sgemm(float *A, float *B, float *C,
          int numARows, int numACols, int numBRows, int numBCols);