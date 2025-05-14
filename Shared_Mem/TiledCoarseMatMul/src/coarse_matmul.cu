// TODO: Implement a tiled matrix multiplication kernel with thread coarsening.
#define TILE_WIDTH 32
#define COARSE_FACTOR 3
//Done!

__global__
void sgemm_coarse(float *A, float *B, float *C, int numARows, int numACols, int numBRows, int numBCols){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the element to work on

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * (TILE_WIDTH * COARSE_FACTOR);

    int numTile  = (numACols+TILE_WIDTH-1)/TILE_WIDTH;

    // Initialize Pvalue
    float Pvalue[COARSE_FACTOR] = {0.0f};

    // Loop over the tiles required to compute the current output value
    for (int ph = 0; ph < numTile; ph++){
	    if(row < numARows && (ph*TILE_WIDTH+tx)< numACols){
		    Mds[ty][tx] = A[row * numACols + (ph * TILE_WIDTH + tx)];
	    }
	    else{
		    Mds[ty][tx] = 0.0f;
	    }
	    __syncthreads();

	    // Load Nds for every Coarse and Compute for B Columns  
	    for(int i=0; i < COARSE_FACTOR; i++){
		    if(ph*TILE_WIDTH+ty < numBRows && (colStart + i *TILE_WIDTH + tx)< numBCols){
		    	Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * numBCols + (colStart + i * TILE_WIDTH + tx)];
		    }
		    else{
		    	Nds[ty][tx] = 0.0f;
		    }
		    __syncthreads();
		    for(int j = 0; j < TILE_WIDTH; j++){
			    Pvalue[i] += Mds[ty][j] * Nds[j][tx];
		    }
		    __syncthreads();
	    }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH + tx;
	if(row < numARows && col < numBCols){
		C[row * numBCols + col] = Pvalue[c];
	}
    }
}

int sgemm(float *A_h, float *B_h, float *C_h, int numARows, int numACols,int numBRows, int numBCols){
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

    sgemm_coarse<<<gridDim,blockDim>>>(A_d, B_d, C_d, numARows, numACols, numBRows, numBCols);

    cudaDeviceSynchronize();
    cudaMemcpy(C_h, C_d, sizeC*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;

}

