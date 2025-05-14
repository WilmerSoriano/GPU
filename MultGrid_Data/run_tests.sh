#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

## COLOR TO GRAYSCALE ##

# Rebuild the project if build/release/rgb2gray and build/release/rgb2gray_datagen do not exist
if [ ! -f build/release/rgb2gray ] || [ ! -f build/release/rgb2gray_datagen ]; then
    cmake --build build 
fi

# Run datagen if data/rgb2gray does not exist
if [ ! -d data/rgb2gray ]; then
    ./build/release/rgb2gray_datagen
fi

echo "********************************"
echo "** Running tests for rgb2gray **"
echo "********************************"

# Loop through directories 0 to 9
for dir in $(seq 0 9)
do
    echo "Running test for directory $dir"
    ./build/release/rgb2gray_test -e data/rgb2gray/$dir/output.pbm -i data/rgb2gray/$dir/input.ppm -t image
done

## BLUR IMAGE ##
# Rebuild the project if build/release/blur and build/release/blur_datagen do not exist
if [ ! -f build/release/blur ] || [ ! -f build/release/blur_datagen ]; then
    cmake --build build 
fi

# Run datagen if data/blur does not exist
if [ ! -d data/blur ]; then
    ./build/release/blur_datagen
fi

echo
echo "****************************"
echo "** Running tests for blur **"
echo "****************************"

# Loop through directories 0 to 9
for dir in $(seq 0 9)
do
    echo "Running test for directory $dir"
    ./build/release/blur_test -e data/blur/$dir/output.ppm -i data/blur/$dir/input.ppm -t image
done

## MATRIX MULTIPLICATION ##

# Rebuild the project if build/release/matrix and build/release/matrix_datagen do not exist
if [ ! -f build/release/matrix ] || [ ! -f build/release/matrix_datagen ]; then
    cmake --build build 
fi

# Run datagen if data/matrix does not exist
if [ ! -d data/matrix ]; then
    ./build/release/matmul_datagen
fi

echo
echo "******************************"
echo "** Running tests for matmul **"
echo "******************************"

# Loop through directories 0 to 8
for dir in $(seq 0 8)
do
    echo "Running test for directory $dir"
    ./build/release/matmul_test -e data/matmul/$dir/output.raw -i data/matmul/$dir/input0.raw,data/matmul/$dir/input1.raw -t matrix
done
