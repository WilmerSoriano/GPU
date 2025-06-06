#include <gputk.h>

static char *base_dir;

static void compute(float *output, float *input0, float *input1, int num) {
    for (int ii = 0; ii < num; ++ii) {
        output[ii] = input0[ii] + input1[ii];
    }
}

static float *generate_data(int n) {
    float *data = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
    }
    return data;
}

static void write_data(char *file_name, float *data, int num) {
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d", num);
    for (int ii = 0; ii < num; ii++) {
        fprintf(handle, "\n%.2f", *data++);
    }
    fflush(handle);
    fclose(handle);
}

static void create_dataset(int datasetNum, int dim) {

    const char *dir_name =
        gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

    char *input0_file_name = gpuTKPath_join(dir_name, "input0.raw");
    char *input1_file_name = gpuTKPath_join(dir_name, "input1.raw");
    char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

    float *input0_data = generate_data(dim);
    float *input1_data = generate_data(dim);
    float *output_data = (float *)calloc(sizeof(float), dim);

    compute(output_data, input0_data, input1_data, dim);

    write_data(input0_file_name, input0_data, dim);
    write_data(input1_file_name, input1_data, dim);
    write_data(output_file_name, output_data, dim);

    free(input0_data);
    free(input1_data);
    free(output_data);
}

int main(int argc, char **argv) {
    // The first program argument should be a 4 digit integer
    // that will be used as the seed for the random number generator.
    if (argc < 2) {
        printf("Usage: %s <seed>\n", argv[0]);
        return 1;
    }

    srand(atoi(argv[1]));

    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "data");

    create_dataset(0, 16);
    create_dataset(1, 64);
    create_dataset(2, 93);
    create_dataset(3, 112);
    create_dataset(4, 1120);
    create_dataset(5, 9921);
    create_dataset(6, 1233);
    create_dataset(7, 1033);
    create_dataset(8, 4096);
    create_dataset(9, 4018);
    return 0;
}