void color_to_grayscale_cpu(unsigned char *output, unsigned char *input, unsigned int y,
                            unsigned int x) {
    for (unsigned int ii = 0; ii < y; ii++) {
        for (unsigned int jj = 0; jj < x; jj++) {
            unsigned int idx = ii * x + jj;
            float r = input[3 * idx];     // red value for pixel
            float g = input[3 * idx + 1]; // green value for pixel
            float b = input[3 * idx + 2];
            output[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}