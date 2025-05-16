#include <iostream>

#include "gputk.h"
#include "conv2d_gpu.h"
#include "utils.hpp"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    float *hostInput, *hostKernel, *hostOutput;
    int inputHeight, inputWidth, kernelSize, outputHeight, outputWidth, padding, stride, dilation;
    float *deviceInput, *deviceKernel, *deviceOutput;

    args = gpuTKArg_read(argc, argv);

    hostInput = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputHeight,
                                    &inputWidth);
    hostKernel = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &kernelSize, &kernelSize);

    // Configure kernel stats
    kernelSize = (int)hostKernel[0];
    padding = (int)hostKernel[1];
    stride = (int)hostKernel[2];
    dilation = (int)hostKernel[3];
    hostKernel += 4;

    outputHeight = computeOutputSize(inputHeight, kernelSize, padding, stride, dilation);
    outputWidth = computeOutputSize(inputWidth, kernelSize, padding, stride, dilation);

    // Allocate the hostC matrix
    hostOutput = (float *)malloc(outputHeight * outputWidth * sizeof(float));

    convolve2D(hostInput, hostKernel, hostOutput, inputHeight, inputWidth, kernelSize, outputHeight, outputWidth, padding, stride, dilation);

    gpuTKSolution(args, hostOutput, outputHeight, outputWidth);

    free(hostOutput);

    return 0;
}