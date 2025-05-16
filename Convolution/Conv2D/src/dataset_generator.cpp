#include "gputk.h"

#include "conv2d_cpu.hpp"
#include "utils.hpp"

static char *base_dir;

static float *generate_data(const unsigned int y,
                            const unsigned int x) {
    unsigned int i;

    const int maxVal = 1;
    float *data = (float *)malloc(y * x * sizeof(float));

    float *p = data;
    for (i = 0; i < y * x; ++i) {
        float val = rand() % RAND_MAX / (float)RAND_MAX * maxVal;
        *p++ = val;
    }

    return data;
}

// Also writes the kernelSize, padding, stride, and dilation to the file
static float *generate_kernel(const unsigned int kernelSize, const unsigned int padding,
                              const unsigned int stride, const unsigned int dilation) {
    unsigned int i;

    const int maxVal = 19;
    float *data = (float *)malloc((kernelSize * kernelSize + 4) * sizeof(float));

    float *p = data;
    p[0] = kernelSize;
    p[1] = padding;
    p[2] = stride;
    p[3] = dilation;
    p+=4;
    for (i = 0; i < kernelSize * kernelSize; ++i) {
        float val = (rand() % maxVal) - 9;
        *p++ = val;
    }

    return data;
}

static void write_data(char *file_name, float *data, unsigned int width,
                       unsigned int height) {
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", height, width);
    for (int ii = 0; ii < height; ii++) {
        for (int jj = 0; jj < width; jj++) {
            fprintf(handle, "%f", *data++);
            if (jj != width - 1) {
                fprintf(handle, " ");
            }
        }
        if (ii != height - 1) {
            fprintf(handle, "\n");
        }
    }

    fflush(handle);
    fclose(handle);
}

static void create_dataset(const int datasetNum, const int y, const int x,
                           const int kernelSize, const int padding,
                           const int stride, const int dilation) {

    const char *dir_name =
        gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

    char *input_file_name = gpuTKPath_join(dir_name, "input.raw");
    char *kernel_file_name = gpuTKPath_join(dir_name, "kernel.raw");
    char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

    float *input_data = generate_data(x, y);
    float *kernel = generate_kernel(kernelSize, padding, stride, dilation);

    unsigned int outputHeight = computeOutputSize(y, kernelSize, padding, stride, dilation);
    unsigned int outputWidth = computeOutputSize(x, kernelSize, padding, stride, dilation);

    printf("Output size: %d x %d\n", outputHeight, outputWidth);

    float *output_data = (float *)malloc(outputHeight * outputWidth * sizeof(float));

    convolve2DCPU(input_data, kernel + 4, output_data, y, x, kernelSize, outputHeight, outputWidth, stride, dilation);

    write_data(input_file_name, input_data, x, y);
    write_data(kernel_file_name, kernel, 1, (kernelSize * kernelSize) + 4);
    write_data(output_file_name, output_data, outputWidth, outputHeight);

    free(input_data);
    free(kernel);
    free(output_data);
}

int main() {

    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "../data");

    create_dataset(0, 4, 4, 3, 1, 1, 1);
    create_dataset(1, 32, 32, 3, 1, 1, 1);
    create_dataset(2, 32, 64, 3, 1, 1, 1);
    create_dataset(3, 17, 34, 3, 1, 1, 1);
    create_dataset(4, 16, 16, 5, 2, 1, 1);
    create_dataset(5, 32, 32, 5, 2, 2, 1);
    create_dataset(6, 32, 64, 5, 2, 3, 1);
    create_dataset(7, 17, 34, 5, 2, 4, 1);
    create_dataset(8, 16, 16, 5, 0, 5, 1);
    create_dataset(9, 32, 32, 5, 0, 1, 1);
    create_dataset(10, 32, 64, 5, 0, 1, 1);
    create_dataset(11, 17, 34, 5, 0, 1, 1);
    create_dataset(12, 16, 16, 5, 1, 1, 2);
    create_dataset(13, 32, 32, 5, 1, 1, 3);
    create_dataset(14, 32, 64, 5, 1, 1, 4);
    create_dataset(15, 17, 34, 5, 1, 1, 2);
    create_dataset(16, 256, 256, 5, 0, 3, 1);
    create_dataset(17, 256, 256, 5, 1, 3, 1);
    create_dataset(18, 256, 256, 5, 2, 3, 1);
    create_dataset(19, 256, 256, 5, 1, 3, 1);
    create_dataset(20, 256, 256, 5, 1, 3, 1);

    return 0;
}
