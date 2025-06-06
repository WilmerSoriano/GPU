#include "gputk.h"

static char *base_dir;

static float compute(float *input, int num) {
    float ret = 0;
    for (int ii = 0; ii < num; ++ii) {
        ret += input[ii];
    }
    return ret;
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

    char *input_file_name = gpuTKPath_join(dir_name, "input.raw");
    char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

    float *input_data = generate_data(dim);
    float output_data;

    output_data = compute(input_data, dim);

    write_data(input_file_name, input_data, dim);
    write_data(output_file_name, &output_data, 1);

    free(input_data);
}

int main() {
    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "../data");
    create_dataset(0, 16);
    create_dataset(1, 64);
    create_dataset(2, 93);
    create_dataset(3, 112);
    create_dataset(4, 1120);
    create_dataset(5, 2233);
    create_dataset(6, 10331);
    create_dataset(7, 40989);
    create_dataset(8, 83482);
    create_dataset(9, 100000);
    return 0;
}
