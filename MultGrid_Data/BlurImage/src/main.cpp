#include "gputk.h"

#include "blur_gpu.h"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    int width = 0;
    int height = 0;
    int channels = 0;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input image> <kernel_size>\n", argv[0]);
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    channels = img.channels();

    cv::Mat img_blurred;

    // Ignore the alpha channel for blurring.
    if (channels > 3) {
        channels = 3;
        img.convertTo(img, CV_32FC3);

        // Create a 3-channel image for the blurred result
        img_blurred = cv::Mat(img.rows, img.cols, CV_32FC3);
    } else {
        img.convertTo(img, CV_32FC1);

        // Create a 1-channel image for the blurred result
        img_blurred = cv::Mat(img.rows, img.cols, CV_32FC1);
    }


    if (img.empty()) {
        fprintf(stderr, "Error: could not load image %s\n", argv[1]);
        return 1;
    }

    float *data_ptr = reinterpret_cast<float *>(img.data);

    // blur the image using the GPU version of blur
    int kernel_size = atoi(argv[2]);

    blur(reinterpret_cast<float *>(img_blurred.data), data_ptr, kernel_size,
         img.cols, img.rows, channels);

    // write the blurred image to disk
    cv::imwrite("output.png", img_blurred);

    return 0;
}
