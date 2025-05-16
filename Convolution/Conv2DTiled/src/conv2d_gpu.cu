#include "conv2d_gpu.h"

#define TILE_SIZE 32
#define KERNEL_LIMIT 5

__constant__ float k_FilterD[KERNEL_LIMIT * KERNEL_LIMIT];

__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight,
    int stride, int dilation) {

    // TODO: Implement the kernel for constant memory filtering
    __shared__ float s_kernel[KERNEL_LIMIT * KERNEL_LIMIT];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int outX = blockIdx.x * TILE_SIZE + tx;
    const int outY = blockIdx.y * TILE_SIZE + ty;
    
    // Load kernel
    if(tx < kernelSize && ty < kernelSize) {
        s_kernel[ty * kernelSize + tx] = kernel[ty * kernelSize + tx];
    }
    __syncthreads();

    if(outX < outputWidth && outY < outputHeight) {
        const int k_radius = kernelSize / 2;
        float sum = 0.0f;
        
        for(int ky = -k_radius; ky <= k_radius; ky++) {
            for(int kx = -k_radius; kx <= k_radius; kx++) {
                // Calculate dilated input indices
                const int inX = outX * stride + kx * dilation;
                const int inY = outY * stride + ky * dilation;
                
                if(inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
                    sum += input[inY * inputWidth + inX] * 
                          s_kernel[(ky + k_radius) * kernelSize + (kx + k_radius)];
                }
            }
        }
        output[outY * outputWidth + outX] = sum;
    }
}

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for shared memory filtering
    __shared__ float inputTile[TILE_SIZE][TILE_SIZE];
    __shared__ float s_kernel[25];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outX = blockIdx.x * TILE_SIZE + tx;
    int outY = blockIdx.y * TILE_SIZE + ty;
    
    // Load input tile
    int inX = outX * stride;
    int inY = outY * stride;
    if(inX < inputWidth && inY < inputHeight) {
        inputTile[ty][tx] = input[inY * inputWidth + inX];
    }
    
    // Load kernel
    if(tx < kernelSize && ty < kernelSize) {
        s_kernel[ty * kernelSize + tx] = kernel[ty * kernelSize + tx];
    }
    __syncthreads();

    if(outX < outputWidth && outY < outputHeight) {
        int k_radius = kernelSize / 2;
        float sum = 0.0f;
        
        for(int ky = -k_radius; ky <= k_radius; ky++) {
            for(int kx = -k_radius; kx <= k_radius; kx++) {
                int x = tx + kx * dilation;
                int y = ty + ky * dilation;
                
                if(x >= 0 && x < TILE_SIZE && y >= 0 && y < TILE_SIZE) {
                    sum += inputTile[y][x] * 
                          s_kernel[(ky + k_radius) * kernelSize + (kx + k_radius)];
                }
            }
        }
        output[outY * outputWidth + outX] = sum;
    }
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