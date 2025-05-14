#include "gputk.h"

#include "matmul_gpu.cuh"

__global__
void sgemm_kernel(float *A, float *B, float *C, int numARows, int numACols, int numBRows, int numBCols) {
    // might need to update function to match the parameter request.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= numARows || col >= numBCols) return;

    //if (row >= numBRows || col >= numACols) return; 
    // TODO: Insert code to implement matrix multiplication here

    float sum = 0.0f;
    for (int k = 0; k < numACols; k++) {
        sum += A[row * numACols + k] * B[k * numBCols + col];
    }
    C[row * numBCols + col] = sum;
    
}

int sgemm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,int numBRows, int numBCols) {
    float *A_d, *B_d, *C_d;

    int sizeA = numARows*numACols;
    int sizeB = numBRows*numBCols;
    int sizeC = numARows*numBCols;

    gpuTKTime_start(GPU, "Allocating GPU memory.");
    // TODO: Allocate GPU memory here
    // Don't forget to wrap the function calls with gpuTKCheck() macro
    
    gpuTKCheck(cudaMalloc(&A_d, sizeA*sizeof(float)));
    gpuTKCheck(cudaMalloc(&B_d, sizeB*sizeof(float)));
    gpuTKCheck(cudaMalloc(&C_d, sizeC*sizeof(float)));

    gpuTKTime_stop(GPU, "Allocating GPU memory.");

    gpuTKTime_start(GPU, "Copying input memory to the GPU.");
    // TODO: Copy memory to the GPU here
    cudaMemcpy(A_d, A_h, sizeA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB*sizeof(float), cudaMemcpyHostToDevice);

    gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

    // TODO: Initialize the grid and block dimensions here
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((numBCols + blockDim.x - 1) / blockDim.x, (numARows + blockDim.y - 1) / blockDim.y, 1);

    gpuTKLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
    gpuTKLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

    gpuTKTime_start(Compute, "Performing CUDA computation");
    // TODO: Launch the GPU Kernel here

    sgemm_kernel<<<gridDim,blockDim>>>(A_d, B_d, C_d, numARows, numACols, numBRows, numBCols);

    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "Performing CUDA computation");

    gpuTKTime_start(Copy, "Copying output memory to the CPU");
    // TODO: Copy the GPU memory back to the CPU here
    cudaMemcpy(C_h, C_d, sizeC*sizeof(float), cudaMemcpyDeviceToHost);

    gpuTKTime_stop(Copy, "Copying output memory to the CPU");

    gpuTKTime_start(GPU, "Freeing GPU Memory");
    // TODO: Free the GPU memory here
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    gpuTKTime_stop(GPU, "Freeing GPU Memory");

    return 0;
}
