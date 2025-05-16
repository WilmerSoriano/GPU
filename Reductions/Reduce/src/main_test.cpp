#include "gputk.h"
#include "reduce.h"

int main(int argc, char **argv) {
    int ii;
    gpuTKArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = gpuTKArg_read(argc, argv);

    hostInput =
        (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1)) {
        numOutputElements++;
    }
    hostOutput = (float *)malloc(numOutputElements * sizeof(float));

    reduce(hostInput, hostOutput, numInputElements);

    /********************************************************************
     * Reduce output vector on the host
     ********************************************************************/
    for(int i = 1; i < numOutputElements; i++){
        hostOutput[0] += hostOutput[i];
    }

    gpuTKSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}