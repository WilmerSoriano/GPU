#include <cmath>

/**
 * Computes the output size of one dimension (height or width) of an image after convolution.
 * 
 * @param inputSize The size of the input dimension (height or width).
 * @param kernelSize The size of the kernel dimension (height or width).
 * @param padding The padding applied to the input dimension.
 * @param stride The stride with which the kernel moves across the input.
 * @param dilation The dilation of the kernel.
 * @return The size of the output dimension after convolution.
 */
int computeOutputSize(int inputSize, int kernelSize, int padding, int stride, int dilation) {
    // Adjust kernel size for dilation
    int effectiveKernelSize = (kernelSize - 1) * dilation + 1;
    
    // Compute output size based on the convolution formula
    int outputSize = std::floor((inputSize - effectiveKernelSize + 2 * padding) / stride) + 1;
    
    return outputSize;
}