#include "conv2d_gpu.h"

#define IN_TILE 32
#define OUT_TILE 34

__constant__ float kFilter_d[25];

__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for constant memory filtering
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(outX >= outputWidth || outY >= outputHeight) return;

    int k_radius = kernelSize / 2;
    float sum = 0.0f;

    for(int ky = -k_radius; ky <= k_radius; ky++) {
        for(int kx = -k_radius; kx <= k_radius; kx++) {
            int inX = outX * stride + kx * dilation;
            int inY = outY * stride + ky * dilation;
            
            if(inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
                const float val = input[inY * inputWidth + inX];
                const float weight = kFilter_d[(ky + k_radius) * kernelSize + (kx + k_radius)];
                sum += val * weight;
            }
        }
    }
    output[outY * outputWidth + outX] = sum;

}

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for shared memory filtering
    __shared__ float s_kernel[25];  // Shared memory for filter
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    // Load filter into shared memory
    if(tid < kernelSize * kernelSize) {
        s_kernel[tid] = kernel[tid];
    }
    __syncthreads();

    const int outX = blockIdx.x * blockDim.x + tx;
    const int outY = blockIdx.y * blockDim.y + ty;
    
    if(outX >= outputWidth || outY >= outputHeight) return;

    const int k_radius = kernelSize / 2;
    float sum = 0.0f;

    for(int ky = -k_radius; ky <= k_radius; ky++) {
        for(int kx = -k_radius; kx <= k_radius; kx++) {
            const int inX = outX * stride + kx * dilation;
            const int inY = outY * stride + ky * dilation;
            
            if(inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
                const float val = input[inY * inputWidth + inX];
                const float weight = s_kernel[(ky + k_radius) * kernelSize + (kx + k_radius)];
                sum += val * weight;
            }
        }
    }
    output[outY * outputWidth + outX] = sum;
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
remover comment and comment constant Kernel*/
//This one worked with shared kernel
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