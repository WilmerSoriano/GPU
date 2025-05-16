#include "gputk.h"
#include "conv2d_gpu.h" // added
#include <opencv2/opencv.hpp>

void compute_gradient_image(float *grad_norms, float *img, int width, int height) {
    // Create sobel filters
    float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Create a 3x3 filter
    float *filter_x = new float[9];
    float *filter_y = new float[9];

    // Copy the sobel filters to the 3x3 filter
    for (int i = 0; i < 9; i++) {
        filter_x[i] = sobel_x[i];
        filter_y[i] = sobel_y[i];
    }

    // Create empty images for the gradient images
    float *grad_x = new float[width * height];
    float *grad_y = new float[width * height];

    // TODO: Call your convolutional kernel to generate the gradient images
    //convolve2D(img, sobel_x, grad_x, height, width, 3, height, width, 1, 1, 1);
    //convolve2D(img, sobel_y, grad_y, height, width, 3, height, width, 1, 1, 1);

    // TODO: Compute the gradient norms
    float max_norm = 0.0f;
    for (int i = 0; i < width * height; ++i) {
        grad_norms[i] = sqrtf(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]);
        if (grad_norms[i] > max_norm) {
            max_norm = grad_norms[i];
        }
    }
    if (max_norm != 0.0f) {
        for (int i = 0; i < width * height; ++i) {
            grad_norms[i] /= max_norm;
        }
    }

    // Cleanup allocated memory
    delete[] filter_x;
    delete[] filter_y;
    delete[] grad_x;
    delete[] grad_y;
}

int main(int argc, char **argv) {
    int width = 0;
    int height = 0;
    int channels = 0;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input image>\n", argv[0]);
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    channels = img.channels();

    cv::Mat img_grad_norms;

    // Ignore the alpha channel
    if (channels >= 3) {
        channels = 3;
        img.convertTo(img, CV_32FC3);

        // Convert the image to grayscale
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        img.convertTo(img, CV_32FC1);
    }

    // Create a 1-channel image for the gradient image
    img_grad_norms = cv::Mat(img.rows, img.cols, CV_32FC1);

    if (img.empty()) {
        fprintf(stderr, "Error: could not load image %s\n", argv[1]);
        return 1;
    }

    float *data_ptr = reinterpret_cast<float *>(img.data);

    compute_gradient_image(reinterpret_cast<float *>(img_grad_norms.data), data_ptr, img.cols, img.rows);

    // write the gradient image to disk
    cv::imwrite("output.png", img_grad_norms);

    return 0;
}