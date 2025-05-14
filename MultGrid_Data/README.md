# Multidimensional-grids and Data

This Repository explores multidimensional data and grids. With a better understanding of how to design kernels and launch configurations for 2D and 3D data.

# Building the Project

To build the project, you will need to use `cmake`. You can build the project with the following commands:

```bash
cmake -B build -DGPUTK_INSTALL_PATH=<path_to_gputk>
cmake --build build
```

For some GPU, remove `-DGPUTK_INSTALL_PATH` flag if not supported.

After completing each part, be sure to rebuild the project with `cmake --build build`.

## Part 1: Converting an Image to Grayscale

The first implementation kernel that converts a color image to grayscale. The Kernel account for multiple channels when using flat indexing. 

### Implementation

In the `ColorToGrayscale` folder, the CPU version is already provided. The Implement of GPU version and call the kernel would be in `main.cpp`.

The following formula convert each pixel in both the CPU and GPU versions:

$$
Y = 0.299 R + 0.587 G + 0.114 B
$$

### Testing

To test the implementation run `bash run_tests.sh` from the base folder. This will test your implementation on a variety of image sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.

### Qualitative Testing

You can test your kernel on a real image by running `rgb2gray`. If compile and runed correctly, this will successfully compile and build an executable in the `build` folder. You can then run this executable with the following command:

```bash
./build/release/rgb2gray <image_path>
```

 you can also compile and run the provided image with the following commands:

```bash
sbatch convert_image.sh
```

If you want to use a different image than the one provided, you can modify the `convert_image.sh` script to use your image instead.

This will produce a new image named `output.png`. The input image must be a PNG file and RGB already. If you want to use a different image format, you will need to modify the code in `main.cpp` to use the appropriate library. The code uses `opencv` to read and write images. If you're running this on your own machine, you will need to install `opencv` first.

## Part 2: Blurring an Image

### Implementation

In the `BlurImage` folder, the CPU version is already provided. The GPU version is added and call the kernel in `main.cpp`. Your GPU version should be able to handle any odd and square size. Additionally, it should work with both grayscale and color images. You can do this by either implementing two kernels or by using a conditional statement in the kernel.

### Testing

To test your implementation, you can run `bash run_tests.sh` from the base folder.  This will test your implementation on a variety of kernel sizes and image sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.

### Qualitative Testing

You can test your kernel on a real image by running `blur`. As long as your kernel was implemented correctly, this will successfully compile and build a `main` executable in the `build` folder. You can then run this executable with the following command:

```bash
./build/release/blur <image_path> <kernel_size>
```

 you can also compile and run the provided image with the following commands(if your having trouble above):

```bash
sbatch blur_image.sh
```

If you want to use a different image than the one provided, you can modify the `blur_image.sh` script to use your image instead.


## Part 3: Matrix Multiplication

For this section I implemented a kernel that performs matrix multiplication.

### Implementation

In the `MatMul` folder, the CPU version is already provided. For Implementation of GPU version and kernel called would be located in `matmul_gpu.cu`. Your GPU version should be able to handle any matrix sizes.

Wrap any calls to CUDA functions with the error checking macro defined in `matmul_gpu.h`. This will automatically check the return value of the CUDA function and print an error message if it fails.

You can rebuild the project with cmake. This will compile the GPU version and the main program that tests it.

### Testing

To test your implementation, you can run `bash run_tests.sh` from the base folder. This will test your implementation on a variety of matrix sizes. The test script will print out the results of each test. If your implementation is correct, all tests should pass.