// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}
#include "scan.h"


__global__ void scanKernel(float *input, float *output, int len, int *blockId, int *flags) {
  __shared__ unsigned int bids;

  if(threadIdx.x == 0){
    bids = atomicAdd(blockId, 1);

    if(bids > 0)
      flags[bids] = 0;
    else
      flags[bids] = 1;
  }
  __syncthreads();

  unsigned int bid = bids;
  unsigned int i = 2*bid*blockDim.x + threadIdx.x;
  __shared__ float A[SECTION_SIZE];

  if( i < len){
    A[threadIdx.x] = input[i];
  }
  
  if(i + blockDim.x < len){
    A[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  }

  for(unsigned int stride =1; stride <= blockDim.x; stride *=2){
    __syncthreads();
    int index = ((threadIdx.x + 1) * 2 * stride) -1;

    if(index < 2*blockDim.x){
      A[index] += A[index - stride];
    }
  }

  for(unsigned int stride = blockDim.x / 2; stride > 0; stride/=2){
    __syncthreads();
    int index = (threadIdx.x+1) * 2 * stride - 1 ;
    
    if(index+stride < 2*blockDim.x){
      A[index+stride] += A[index];
    }
    __syncthreads();
  }

  if( i < len){
    output[i] = A[threadIdx.x];
  }
  
  if(i + blockDim.x < len){
    output[i + blockDim.x] = A[threadIdx.x + blockDim.x];
  }

  __syncthreads();
  __shared__ float addVal;

  if(threadIdx.x == BLOCK_SIZE-1){
    addVal = 0;

    while(atomicAdd(&flags[bids], 0) == 0);

    if(bid != 0){
      addVal = output[2*bid*blockDim.x-1];
      if(i < len){
        output[i] = output[i] + addVal;
      }
      if(i + blockDim.x< len){
        output[i + blockDim.x] = output[i + blockDim.x] + addVal;
      }
    }
    __threadfence();// NEW NEW NEW!!! study later...

    atomicAdd(&flags[bid +1], 1);
  }
  __syncthreads();

  if(threadIdx.x != BLOCK_SIZE -1){
    if(i < len){
      output[i] = output[i] + addVal;
    }

    if(i + blockDim.x < len){
      output[i + blockDim.x] = output[i + blockDim.x] + addVal;
    }
  }
}
 
void scan(float *input, float *output, int len) {
  float *inputD;
  float *outputD;
  int blockId= 0;
  int *blockId_D;
  int *flag_d;

  dim3 blockSize(BLOCK_SIZE,1, 1);
  dim3 gridSize((len+2*blockSize.x-1)/(2*blockSize.x),1,1);

  cudaMalloc(&inputD, sizeof(float)*len);
  cudaMalloc(&outputD, sizeof(float)*len);
  cudaMalloc(&blockId_D, sizeof(int));
  cudaMalloc(&flag_d, sizeof(int) * (MAXDIMX*MAXDIMY + 2*BLOCK_SIZE -1)/ (2 * BLOCK_SIZE));

  cudaMemcpy(inputD, input, sizeof(float)*len, cudaMemcpyHostToDevice);
  cudaMemcpy(blockId_D, &blockId, sizeof(int), cudaMemcpyHostToDevice);

  scanKernel<<<gridSize, blockSize>>>(inputD, outputD, len, blockId_D, flag_d);

  cudaMemcpy(output, outputD, sizeof(float)*len, cudaMemcpyDeviceToHost);

  cudaFree(inputD);
  cudaFree(outputD);
  cudaFree(blockId_D);
  cudaFree(flag_d);
}
