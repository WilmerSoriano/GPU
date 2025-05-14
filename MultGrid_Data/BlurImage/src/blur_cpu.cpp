void blur_cpu(unsigned char *out, unsigned char *in, int size,
              unsigned int height, unsigned int width) {
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            int pixVal = 0;
            int pixels = 0;
            // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurrow = -size; blurrow < size + 1; ++blurrow) {
                for (int blurcol = -size; blurcol < size + 1; ++blurcol) {
                    int currow = row + blurrow;
                    int curcol = col + blurcol;
                    // Verify we have a valid image pixel
                    if (currow > -1 && static_cast<unsigned int>(currow) < height &&
                        curcol > -1 && static_cast<unsigned int>(curcol) < width) {
                        pixVal += in[currow * width + curcol];
                        pixels++; // Keep track of number of pixels in the avg
                    }
                }
            }

            // Write our new pixel value out
            out[row * width + col] = (unsigned char)(pixVal / pixels);
        }
    }
}