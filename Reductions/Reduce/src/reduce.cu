// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include "reduce.h"


__global__ void reduceKernel(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ float inputS[BLOCK_SIZE];
  int outputLen = len / (BLOCK_SIZE << 1);

  if(len % (BLOCK_SIZE << 1)){
    outputLen++;
  }

  unsigned int segment = 2*blockDim.x * blockIdx.x;
  unsigned int t = threadIdx.x;
  unsigned int i = segment + t; 

  if(i < len && i + BLOCK_SIZE < len)
    inputS[t] = input[i] + input[i + BLOCK_SIZE];
  else if(i < len)
    inputS[t] = input[i];
  else
    inputS[t] = 0;

  //@@ Traverse the reduction tree
  for(unsigned int stride = blockDim.x/2; stride >= 1; stride/=2){
    __syncthreads();
    if(t < stride && t + stride < len)
      inputS[t] += inputS[t + stride];
  }
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  if(t == 0){
    if(blockIdx.x < outputLen){
      *(output + blockIdx.x) = inputS[0];
    }
  }
}

void reduce(float *input, float *output, int len) {
    float *inputD, *outputD;
    int output_len = len / (BLOCK_SIZE << 1);

    if(len % (BLOCK_SIZE << 1))
      output_len++;

    //@@ Allocate GPU memory here
    cudaMalloc((void**)&inputD, len*sizeof(float));
    cudaMalloc((void**)&outputD, output_len*sizeof(float));

    //@@ Copy memory to the GPU here
    cudaMemcpy(inputD, input, len*sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(output_len);

    //@@ Launch the GPU Kernel here
    reduceKernel<<<dimGrid, dimBlock>>>(inputD, outputD, len);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(output, outputD, output_len*sizeof(float), cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(inputD);
    cudaFree(outputD);
}
