#pragma once

// Error checking macro
#define gpuTKCheck(stmt)                                                       \
    do {                                                                       \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                     \
            return -1;                                                         \
        }                                                                      \
    } while (0)

// sgemm stands for "single-precision general matrix multiply"
// This follows the naming convention of the BLAS library
int sgemm(float *A, float *B, float *C,
          int numARows, int numACols, int numBRows, int numBCols);