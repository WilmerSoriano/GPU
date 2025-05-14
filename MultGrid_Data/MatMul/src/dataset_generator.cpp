#include "gputk.h"

#include "matmul_cpu.hpp"

static char *base_dir;

static float *generate_data(int height, int width) {
    float *data = (float *)malloc(sizeof(float) * width * height);
    int i;
    for (i = 0; i < width * height; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
    return data;
}

static void write_data(char *file_name, float *data, int height, int width) {
    int ii, jj;
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", height, width);
    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            fprintf(handle, "%.2f", *data++);
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

static void write_transpose_data(char *file_name, float *data, int height,
                                 int width) {
    int ii, jj;
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", width, height);
    for (jj = 0; jj < width; jj++) {
        for (ii = 0; ii < height; ii++) {
            fprintf(handle, "%.2f", data[ii * width + jj]);
            if (ii != height - 1) {
                fprintf(handle, " ");
            }
        }
        if (jj != width - 1) {
            fprintf(handle, "\n");
        }
    }
    fflush(handle);
    fclose(handle);
}

static void create_dataset(int datasetNum, int numARows, int numACols,
                           int numBCols) {
    int numBRows = numACols;
    int numCRows = numARows;
    int numCCols = numBCols;

    const char *dir_name =
        gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

    char *input0_file_name = gpuTKPath_join(dir_name, "input0.raw");
    char *input1_file_name = gpuTKPath_join(dir_name, "input1.raw");
    char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

    float *input0_data = generate_data(numARows, numACols);
    float *input1_data = generate_data(numBRows, numBCols);
    float *output_data = (float *)calloc(sizeof(float), numCRows * numCCols);

    sgemm_cpu(output_data, input0_data, input1_data, numARows, numACols, numBRows,
              numBCols, numCRows, numCCols);

    write_data(input0_file_name, input0_data, numARows, numACols);
    write_data(input1_file_name, input1_data, numBRows, numBCols);
    write_data(output_file_name, output_data, numCRows, numCCols);

    free(input0_data);
    free(input1_data);
    free(output_data);
}

int main() {
    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "data/matmul");

    create_dataset(0, 16, 16, 16);
    create_dataset(1, 64, 64, 64);
    create_dataset(2, 64, 128, 64);
    create_dataset(3, 112, 48, 16);
    create_dataset(4, 84, 84, 84);
    create_dataset(5, 80, 99, 128);
    create_dataset(6, 67, 53, 64);
    create_dataset(7, 29, 117, 85);
    create_dataset(8, 191, 19, 241);
    return 0;
}
