#include <stdlib.h>
#include <stdio.h>

#include "matmul_gpu.h"


static float *generate_data(int height, int width) {
    float *data = (float *)malloc(sizeof(float) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
    return data;
}


int main(int argc, char **argv) {
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numACols; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBCols; // number of columns in the matrix B
    int numCRows;
    int numCCols;

    if (argc != 4) {
        printf("Usage: %s A_rows A_columns B_columns\n", argv[0]);
        return 1;
    }

    // Allocate and initialize the matrices
    numARows = atoi(argv[1]);
    numACols = atoi(argv[2]);
    numBRows = numACols;
    numBCols = atoi(argv[3]);

    hostA = generate_data(numARows, numACols);
    hostB = generate_data(numACols, numBCols);

    // Allocate the hostC matrix
    hostC = (float *)malloc(numARows * numBCols * sizeof(float));

    // Compute the size of matrix C
    numCRows = numARows;
    numCCols = numBCols;

    sgemm(hostA, hostB, hostC, numARows, numACols, numBRows, numBCols);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}