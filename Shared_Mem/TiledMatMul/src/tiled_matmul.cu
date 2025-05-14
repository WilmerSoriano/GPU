#define TILE_WIDTH 32

__global__
void sgemm_tiled(float *A, float *B, float *C, int numARows, int numACols, int numBRows, int numBCols) {
    // TODO: Insert code to implement tiled matrix multiplication

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
   

    // Identify the row and column of the C element to work on
    // blockDim.y and block.x
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    // n or width = matrix size   
    int Width =(numACols+TILE_WIDTH-1)/ TILE_WIDTH;

    for(int ph = 0; ph < Width; ++ph) {
    // Collaborative loading of A and B tiles into shared memory
        if (Row < numARows && ph * TILE_WIDTH + tx < numACols) {
		//Width + ph*TILE_WIDTH+tx
            Mds[ty][tx] = A[Row * numACols+ (ph*TILE_WIDTH + tx)];
        } else {
            Mds[ty][tx] = 0.0;
        }
        if (ph * TILE_WIDTH + ty < numACols && Col < numBCols) {
            Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBCols + Col];
        } else {
            Nds[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
	__syncthreads();
    }
    if (Row < numARows && Col < numBCols){
	    C[Row * numBCols + Col] = Pvalue;
    }
}


int sgemm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,int numBRows, int numBCols) {
    float *A_d, *B_d, *C_d;

    int sizeA = numARows*numACols;
    int sizeB = numBRows*numBCols;
    int sizeC = numARows*numBCols;
    
    cudaMalloc(&A_d, sizeA*sizeof(float));
    cudaMalloc(&B_d, sizeB*sizeof(float));
    cudaMalloc(&C_d, sizeC*sizeof(float));

    cudaMemcpy(A_d, A_h, sizeA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((numBCols + blockDim.x - 1) / blockDim.x, (numARows + blockDim.y - 1) / blockDim.y, 1);

    sgemm_tiled<<<gridDim,blockDim>>>(A_d, B_d, C_d, numARows, numACols, numBRows, numBCols);

    cudaDeviceSynchronize();
    cudaMemcpy(C_h, C_d, sizeC*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
