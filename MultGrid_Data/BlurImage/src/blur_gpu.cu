#include "blur_gpu.h"

#include "gputk.h"

__global__
void blurKernel(float *out, float *in, int size, int width, int height) {
    // TODO: Complete CUDA Kernel for image blurring.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // width = Column and height = Row
    // blur size is needed for range boundry for neighboring calculation.

    float sum = 0.0f;
    int numPixels = 0;

    int range = (size-1)/2;        // <=This Give us the range : example blur size is 5x5, range must be [-2,2]

    for (int ky = -size; ky <= size; ky++) {
        for (int kx = -size; kx <= size; kx++) {
            if (x + kx < 0 || x + kx >= width) continue;
            if (y + ky < 0 || y + ky >= height) continue;
            int index = (y + ky) * width + (x + kx);
            sum += in[index];
            numPixels++;
        }
    }
    out[y * width + x] = sum / numPixels;
}

__global__
void blurColorKernel(float *out, float *in, int size, int width, int height, int channels){
    // TODO: Complete CUDA Kernel for image blurring.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // width = Column and height = Row
    // blur size is needed for range boundry for neighboring calculation.

   // int range = (size-1)/2;        // <=This Give us the range : example blur size is 5x5, range must be [-2,2]

    for(int A = 0; A < channels;A++){
        float sum = 0.0f;
        int numPixels = 0;

        for (int ky = -size; ky <= size; ky++) {
            for (int kx = -size; kx <= size; kx++) {
                if (x + kx < 0 || x + kx >= width) continue;
            	if (y + ky < 0 || y + ky >= height) continue;
            	int index = ((y + ky) * width + (x + kx)) * channels + A;
            	sum += in[index];
            	numPixels++;
            }
    	}
    	out[(y * width + x) * channels + A] = sum / numPixels;
    }
}

void blur(float *out_h, float *in_h, int size, int width, int height, int channels) {
    float *deviceInputImageData = nullptr;
    float *deviceOutputImageData = nullptr;

    gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

    gpuTKTime_start(GPU, "Doing GPU memory allocation");
    // TODO: Allocate device memory
    cudaMalloc(&deviceInputImageData, width*height*channels*sizeof(float));
    cudaMalloc(&deviceOutputImageData, width*height*channels*sizeof(float));

    gpuTKTime_stop(GPU, "Doing GPU memory allocation");

    gpuTKTime_start(Copy, "Copying data to the GPU");
    // TODO: Copy data to device

    cudaMemcpy(deviceInputImageData, in_h, width*height*channels*sizeof(float), cudaMemcpyHostToDevice); //Is the size correct?

    gpuTKTime_stop(Copy, "Copying data to the GPU");

    // TODO: Set up block and grid sizes
    dim3 blockSize(32,32);//256 threads
    dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);

    gpuTKTime_start(Compute, "Doing the computation on the GPU");
    // TODO: Launch kernel
    if(channels == 1){
       blurKernel<<<gridSize, blockSize>>>(deviceOutputImageData, deviceInputImageData, size, width, height);
    }
    else{
       blurColorKernel<<<gridSize, blockSize>>>(deviceOutputImageData, deviceInputImageData, size, width, height, channels);
    }

    cudaDeviceSynchronize(); 
    gpuTKTime_stop(Compute, "Doing the computation on the GPU");

    gpuTKTime_start(Copy, "Copying data from the GPU");
    // TODO: Copy data back to host
    cudaMemcpy(out_h, deviceOutputImageData, width*height*channels*sizeof(float), cudaMemcpyDeviceToHost); 

    gpuTKTime_stop(Copy, "Copying data from the GPU");

    // TODO: Free device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

}
