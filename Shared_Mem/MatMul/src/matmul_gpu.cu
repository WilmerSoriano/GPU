#include "gputk.h"

#include "matmul_gpu.h"

#define TILE_WIDTH 32

__global__
void sgemm_kernel(float *A, float *B, float *C, int numARows,
                  int numACols, int numBRows, int numBCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numBCols) {
        float sum = 0;
        for (int ii = 0; ii < numACols; ii++) {
            sum += A[row * numACols + ii] * B[ii * numBCols + col];
        }
        C[row * numBCols + col] = sum;
    }
}

int sgemm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,
           int numBRows, int numBCols) {
    float *A_d, *B_d, *C_d;

    // Allocate memory
    gpuErrchk(
        cudaMalloc((void **)&A_d, numARows * numACols * sizeof(float)));
    gpuErrchk(
        cudaMalloc((void **)&B_d, numBRows * numBCols * sizeof(float)));
    gpuErrchk(
        cudaMalloc((void **)&C_d, numARows * numBCols * sizeof(float)));

    // Copy data to device
    gpuErrchk(cudaMemcpy(A_d, A_h,
                          numARows * numACols * sizeof(float),
                          cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(B_d, B_h,
                          numBRows * numBCols * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Execute the kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((numBCols + dimBlock.x - 1) / dimBlock.x,
                 (numARows + dimBlock.y - 1) / dimBlock.y);
    sgemm_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, numARows, numACols,
                                       numBRows, numBCols);
    gpuErrchk(cudaDeviceSynchronize());

    // Copy data back to host
    gpuErrchk(cudaMemcpy(C_h, C_d,
                          numARows * numBCols * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}