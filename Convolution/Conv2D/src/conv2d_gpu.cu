#include "conv2d_gpu.h"

#define IN_TILE 32
#define OUT_TILE 34

__constant__ float kFilter_d[25];

// SELF NOTE: both kernel will not be using padding
__global__ void convolve2D_constant_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {

    // TODO: Implement the kernel for constant memory filtering
    int out_Xcol = blockIdx.x * blockDim.x + threadIdx.x;
    int out_Yrow = blockIdx.y * blockDim.y + threadIdx.y;
    int kernelRadius = kernelSize / 2;

    if(out_Xcol >= outputWidth || out_Yrow >= outputHeight)
	    return;

    float sum = 0.0f;

    for(int i = -kernelRadius; i <= kernelRadius; i++){
	    for(int j = -kernelRadius; j <= kernelRadius; j++){
		    int index_X = out_Xcol * stride + j * dilation; //+dilation * kernelRadius;
		    int index_Y = out_Yrow * stride + i * dilation; //+dilation * kernelRadius;

		    if(index_X >= 0 && index_X < inputWidth && index_Y >= 0 && index_Y < inputHeight){
			    float inputOffset = input[index_Y * inputWidth + index_X];
			    float kernelOffset = kernel[(i + kernelRadius) * kernelSize + (j +kernelRadius)];
			    sum += inputOffset * kernelOffset;
		    }
	    }
    }

    int outputOffset = out_Yrow * outputWidth + out_Xcol;
    output[outputOffset] = sum;
}

__global__ void convolve2D_shared_kernel(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, int inputHeight, 
    int kernelSize, int outputWidth, int outputHeight, 
    int stride, int dilation) {
    
    __shared__ float s_input[OUT_TILE][OUT_TILE]; // +2 for padding
    __shared__ float s_kernel[5][5];   

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int outX = blockIdx.x * blockDim.x + tx;
    const int outY = blockIdx.y * blockDim.y + ty;
    
    const int k_radius = kernelSize / 2;
    const int inStartX = blockIdx.x * blockDim.x * stride - k_radius * dilation;
    const int inStartY = blockIdx.y * blockDim.y * stride - k_radius * dilation;

    for(int dy = 0; dy < OUT_TILE; dy += IN_TILE) {
        for(int dx = 0; dx < OUT_TILE; dx += IN_TILE) {
            const int loadX = inStartX + tx + dx;
            const int loadY = inStartY + ty + dy;
            
            if(loadX >= 0 && loadX < inputWidth && loadY >= 0 && loadY < inputHeight) {
                s_input[ty + dy][tx + dx] = input[loadY * inputWidth + loadX];
            } else {
                s_input[ty + dy][tx + dx] = 0.0f;
            }
        }
    }

    if(tx < kernelSize && ty < kernelSize) {
        s_kernel[ty][tx] = kernel[ty * kernelSize + tx];
    }
    
    __syncthreads();

    if(outX < outputWidth && outY < outputHeight) {
        float sum = 0.0f;
        const int baseX = tx * stride + k_radius * dilation;
        const int baseY = ty * stride + k_radius * dilation;

        for(int ky = -k_radius; ky <= k_radius; ky++) {
            for(int kx = -k_radius; kx <= k_radius; kx++) {
                const int x = baseX + kx * dilation;
                const int y = baseY + ky * dilation;
                
                if(x >= 0 && x < 34 && y >= 0 && y < 34) {
                    sum += s_input[y][x] * s_kernel[ky + k_radius][kx + k_radius];
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
remover comment and comment constant Kernel
    convolve2D_shared_kernel<<<blocksDim,threadsDim>>>(
		    deviceInput, deviceKernel, deviceOutput, 
		    inputWidth, inputHeight,kernelSize, outputWidth, 
		    outputHeight, stride, dilation);
*/
    convolve2D_constant_kernel<<<blocksDim,threadsDim>>>(
                    deviceInput, deviceKernel, deviceOutput,
                    inputWidth, inputHeight,kernelSize, outputWidth,
                    outputHeight, stride, dilation);
/**/
    cudaMemcpy(output, deviceOutput, outputHW*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceKernel);
    cudaFree(deviceOutput);
}
