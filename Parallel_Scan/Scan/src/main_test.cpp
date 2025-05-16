#include "gputk.h"
#include "scan.h"

int main(int argc, char **argv) {
    int ii;
    gpuTKArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numElements;  // number of elements in the input list

    args = gpuTKArg_read(argc, argv);

    hostInput =
        (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));

    scan(hostInput, hostOutput, numElements);

    gpuTKSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}