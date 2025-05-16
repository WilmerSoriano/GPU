#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define MAXDIMX 2048
#define MAXDIMY 65535

#define SECTION_SIZE 2 * BLOCK_SIZE

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

void scan(float *input, float *output, int len);