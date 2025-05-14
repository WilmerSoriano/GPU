#include "gputk.h"

#include "tiled_matmul.h"

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

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                                 &numACols);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                                 &numBCols);
    // Allocate the hostC matrix
    hostC = (float *)malloc(numARows * numBCols * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    // TODO: Compute the size of matrix C
    numCRows = numARows;
    numCCols = numBCols;

    gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numACols);
    gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBCols);
    gpuTKLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCCols);

    sgemm(hostA, hostB, hostC, numARows, numACols, numBRows, numBCols);

    gpuTKSolution(args, hostC, numARows, numBCols);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}