/**
 * Performs a 2D convolution operation on the input image using the given kernel.
 *
 * @param input The input image data.
 * @param kernel The convolution kernel.
 * @param output The output image data.
 * @param inputHeight The height of the input image.
 * @param inputWidth The width of the input image.
 * @param kernelSize The size of the convolution kernel.
 * @param outputHeight The height of the output image.
 * @param outputWidth The width of the output image.
 * @param stride The stride value for the convolution operation.
 * @param dilation The dilation value for the convolution operation.
 */

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
    unsigned int dilation) {

    int kernelRadius = kernelSize / 2;

    // Assuming input and kernel are already padded as necessary
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sum = 0;
            for (int m = -kernelRadius; m <= kernelRadius; ++m) {
                for (int n = -kernelRadius; n <= kernelRadius; ++n) {
                    // Calculate the position in the input, considering stride, and dilation
                    int ix = x * stride + n * dilation;
                    int iy = y * stride + m * dilation;

                    // Check bounds
                    if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                        int inputOffset =
                            iy * inputWidth + ix;
                        int kernelOffset = (m + kernelRadius) * kernelSize +
                                           (n + kernelRadius);
                        // Perform multiplication and accumulate
                        sum += input[inputOffset] * kernel[kernelOffset];
                    }
                }
            }
            // Write the computed value to the output matrix
            output[y * outputWidth + x] = sum;
        }
    }
}
