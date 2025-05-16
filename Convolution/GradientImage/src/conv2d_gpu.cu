#include "conv2d_gpu.h"

#define IN_TILE_DIM 32
#define FILTER_RADIUS 1  // For 3x3 kernel (radius = 2)
#define FILTER_DIM (2 * FILTER_RADIUS + 1)
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float kFilter_d[FILTER_DIM][FILTER_DIM];

__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for constant memory filtering
    __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
    
    // Input tile coordinates (with halo)
    int col = blockIdx.x * OUT_TILE_DIM * stride + threadIdx.x * stride - FILTER_RADIUS * dilation;
    int row = blockIdx.y * OUT_TILE_DIM * stride + threadIdx.y * stride - FILTER_RADIUS * dilation;
    
    // Load input tile into shared memory (with zero-padding)
    if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
        inputTile[threadIdx.y][threadIdx.x] = input[row * inputWidth + col];
    } else {
        inputTile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Check if output position is valid (excluding halo)
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    if (outRow < 0 || outRow >= outputHeight || outCol < 0 || outCol >= outputWidth) return;

    float sum = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int tileRow = threadIdx.y + fRow - FILTER_RADIUS;
            int tileCol = threadIdx.x + fCol - FILTER_RADIUS;
            
            if (tileRow >= 0 && tileRow < IN_TILE_DIM && tileCol >= 0 && tileCol < IN_TILE_DIM) {
                // Access shared memory
                sum += inputTile[tileRow][tileCol] * kFilter_d[fRow][fCol];
            } else {
                // Access global memory with dilation
                int inRow = row + fRow * dilation;
                int inCol = col + fCol * dilation;
                if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
                    sum += input[inRow * inputWidth + inCol] * kFilter_d[fRow][fCol];
                }
            }
        }
    }
    
    output[outRow * outputWidth + outCol] = sum;
}

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for shared memory filtering
    __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float s_kernel[FILTER_DIM][FILTER_DIM];
    
    // Load kernel into shared memory
    if (threadIdx.x < FILTER_DIM && threadIdx.y < FILTER_DIM) {
        s_kernel[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * FILTER_DIM + threadIdx.x];
    }
    
    // Input tile coordinates (with halo)
    int col = blockIdx.x * OUT_TILE_DIM * stride + threadIdx.x * stride - FILTER_RADIUS * dilation;
    int row = blockIdx.y * OUT_TILE_DIM * stride + threadIdx.y * stride - FILTER_RADIUS * dilation;
    
    // Load input tile into shared memory (with zero-padding)
    if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
        inputTile[threadIdx.y][threadIdx.x] = input[row * inputWidth + col];
    } else {
        inputTile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Check if output position is valid (excluding halo)
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    if (outRow < 0 || outRow >= outputHeight || outCol < 0 || outCol >= outputWidth) return;

    float sum = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int tileRow = threadIdx.y + fRow - FILTER_RADIUS;
            int tileCol = threadIdx.x + fCol - FILTER_RADIUS;
            
            if (tileRow >= 0 && tileRow < IN_TILE_DIM && tileCol >= 0 && tileCol < IN_TILE_DIM) {
                // Access shared memory
                sum += inputTile[tileRow][tileCol] * s_kernel[fRow][fCol];
            } else {
                // Access global memory with dilation
                int inRow = row + fRow * dilation;
                int inCol = col + fCol * dilation;
                if (inRow >= 0 && inRow < inputHeight && inCol >= 0 && inCol < inputWidth) {
                    sum += input[inRow * inputWidth + inCol] * s_kernel[fRow][fCol];
                }
            }
        }
    }
    
    output[outRow * outputWidth + outCol] = sum;
}

void convolve2D(
    const float *input,
    const float *kernel,
    float *output,
    unsigned int inputHeight,
    unsigned int inputWidth,
    unsigned int kernelSize,
    unsigned int outputHeight,
    unsigned int outputWidth,
    unsigned int padding,
    unsigned int stride,
    unsigned int dilation) {

    // Allocate device memory
    float *deviceInput, *deviceKernel, *deviceOutput;

    // TODO: Complete host function.
    unsigned int inputHW = inputHeight*inputWidth;
    cudaMalloc(&deviceInput, inputHW*sizeof(float));

    unsigned int kern2Size = kernelSize*kernelSize;
    cudaMalloc(&deviceKernel, kern2Size*sizeof(float));

    unsigned int outputHW = outputHeight*outputWidth;
    cudaMalloc(&deviceOutput, outputHW*sizeof(float));

    cudaMemcpy(deviceInput, input, inputHW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernel, kernel, kern2Size*sizeof(float), cudaMemcpyHostToDevice);
    // TODO: Complete host function.

    dim3 threadsDim(32,32);
    dim3 blocksDim((outputWidth+threadsDim.x-1)/threadsDim.x, (outputHeight+threadsDim.y-1)/threadsDim.y);
/* 
HERE IS THE SHARED MEMORY VERSION !!!
remover comment when needed and add comment if use diff kernel*/
    convolve2D_shared_kernel<<<blocksDim,threadsDim>>>(
		    deviceInput, deviceKernel, deviceOutput, 
		    inputWidth, inputHeight,kernelSize, outputWidth, 
		    outputHeight, stride, dilation);
/*
    convolve2D_constant_kernel<<<blocksDim,threadsDim>>>(
                    deviceInput, deviceKernel, deviceOutput,
                    inputWidth, inputHeight,kernelSize, outputWidth,
                    outputHeight, stride, dilation);
*/
    cudaMemcpy(output, deviceOutput, outputHW*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceKernel);
    cudaFree(deviceOutput);
}