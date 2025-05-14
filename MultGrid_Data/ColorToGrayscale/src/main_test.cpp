#include "gputk.h"

#include "color_to_grayscale_gpu.h"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    gpuTKImage_t inputImage;
    gpuTKImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;

    args = gpuTKArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = gpuTKArg_getInputFile(args, 0);

    inputImage = gpuTKImport(inputImageFile);

    imageWidth = gpuTKImage_getWidth(inputImage);
    imageHeight = gpuTKImage_getHeight(inputImage);

    // Since the image is monochromatic, it only contains one channel
    outputImage = gpuTKImage_new(imageWidth, imageHeight, 1);

    hostInputImageData = gpuTKImage_getData(inputImage);
    hostOutputImageData = gpuTKImage_getData(outputImage);

    colorToGrayscale(hostOutputImageData, hostInputImageData, imageHeight, imageWidth);

    gpuTKSolution(args, outputImage);

    gpuTKImage_delete(outputImage);
    gpuTKImage_delete(inputImage);

    return 0;
}