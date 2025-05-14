#include "gputk.h"

#include "coarse_matmul.h"

int main(int argc, char **argv) {
    gpuTKArg_t args;
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

    args = gpuTKArg_read(argc, argv);

    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                                 &numACols);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                                 &numBCols);

    // Allocate the hostC matrix
    hostC = (float *)malloc(numARows * numBCols * sizeof(float));

    // TODO: Call your GPU function here
    sgemm(hostA, hostB, hostC, numARows, numACols, numBRows, numBCols);

    gpuTKSolution(args, hostC, numARows, numBCols);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
