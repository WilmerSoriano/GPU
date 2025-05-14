#include "gputk.h"

#include "color_to_grayscale_gpu.h"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    int width = 0;
    int height = 0;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input image>\n", argv[0]);
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    img.convertTo(img, CV_32FC3);
    cv::Mat img_grayscale(img.rows, img.cols, CV_32FC1);

    if (img.empty()) {
        fprintf(stderr, "Error: could not load image %s\n", argv[1]);
        return 1;
    }

    float *data_ptr = reinterpret_cast<float *>(img.data);

    colorToGrayscale(reinterpret_cast<float *>(img_grayscale.data), data_ptr,
                     img.cols, img.rows);

    // write the blurred image to disk
    cv::imwrite("output.png", img_grayscale);

    return 0;
}