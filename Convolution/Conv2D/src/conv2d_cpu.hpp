void convolve2DCPU(
    const float *input,
    const float *kernel,
    float *output,
    unsigned int inputHeight,
    unsigned int inputWidth,
    unsigned int kernelSize,
    unsigned int outputHeight,
    unsigned int outputWidth,
    unsigned int stride,
    unsigned int dilation);
