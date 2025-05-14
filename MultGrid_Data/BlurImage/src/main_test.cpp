/*
 * Part 2: Blurring an Image
 * 
 * This file is for testing on data generated from `dataset_generator.cpp`.
 * You do not need to edit anything in this file as long as you have implemented
 * the `blur` function in `blur_gpu.cu`.
 */

#include "gputk.h"

#include "blur_gpu.h"

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

    // The input image is in grayscale, so the number of channels
    // is 1
    imageWidth = gpuTKImage_getWidth(inputImage);
    imageHeight = gpuTKImage_getHeight(inputImage);

    // Since the image is monochromatic, it only contains only one channel
    outputImage = gpuTKImage_new(imageWidth, imageHeight, 1);

    hostInputImageData = gpuTKImage_getData(inputImage);
    hostOutputImageData = gpuTKImage_getData(outputImage);

    blur(hostOutputImageData, hostInputImageData, 5, imageWidth, imageHeight, 1);

    gpuTKSolution(args, outputImage);

    gpuTKImage_delete(outputImage);
    gpuTKImage_delete(inputImage);

    return 0;
}
