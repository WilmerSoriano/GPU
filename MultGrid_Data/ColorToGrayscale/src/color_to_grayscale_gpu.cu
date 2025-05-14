#include "gputk.h"

#include "color_to_grayscale_gpu.h"

__global__
void colorToGrayscale_kernel(float *output, float *input, int width, int height) {
    // TODO: Implement Kernel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int index = y * width + x; // Changed the name and types to match the function.
    int inputOffset = index * 3;
    float r = input[inputOffset];
    float g = input[inputOffset + 1];
    float b = input[inputOffset + 2];
    float channelSum = 0.299f * r + 0.587f * g + 0.114f * b;
    output[index] = channelSum;
}

void colorToGrayscale(float *output, float *input, int width, int height) {
    float *deviceInputImageData = nullptr;
    float *deviceOutputImageData = nullptr;

    gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

    gpuTKTime_start(GPU, "Doing GPU memory allocation");
    // TODO: Allocate GPU memory here
    
    cudaMalloc(&deviceInputImageData, width*height*3*sizeof(float));
    cudaMalloc(&deviceOutputImageData, width*height*sizeof(float));

    gpuTKTime_stop(GPU, "Doing GPU memory allocation");

    gpuTKTime_start(Copy, "Copying data to the GPU");
    // TODO: Copy data to GPU here
    cudaMemcpy(deviceInputImageData, input, width*height*3*sizeof(float), cudaMemcpyHostToDevice);

    gpuTKTime_stop(Copy, "Copying data to the GPU");

    ///////////////////////////////////////////////////////
    gpuTKTime_start(Compute, "Doing the computation on the GPU");
    // TODO: Configure launch parameters and call kernel
    dim3 blockSize(32,32,1);//threads
    dim3 gridSize(ceil((float)width/32), ceil((float)height/32)); //blocks
    colorToGrayscale_kernel<<<gridSize, blockSize>>>(deviceOutputImageData, deviceInputImageData, width, height);

    gpuTKTime_stop(Compute, "Doing the computation on the GPU");

    ///////////////////////////////////////////////////////
    gpuTKTime_start(Copy, "Copying data from the GPU");
    // TODO: Copy data from GPU here
    cudaMemcpy(output, deviceOutputImageData, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    gpuTKTime_stop(Copy, "Copying data from the GPU");

    gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // TODO: Free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

}
