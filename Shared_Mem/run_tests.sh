#!/bin/bash

## MATRIX MULTIPLICATION ##

cmake --build build 

# Run datagen if data/matrix does not exist
if [ ! -d data/matmul ]; then
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

## TILED MATRIX MULTIPLICATION ##

# Run datagen if data/tiled_matrix does not exist
if [ ! -d data/matmul_tiled ]; then
    ./build/release/tiledmatmul_datagen
fi

echo
echo "************************************"
echo "** Running tests for tiled matmul **"
echo "************************************"

# Loop through directories 0 to 8
for dir in $(seq 0 8)
do
    echo "Running test for directory $dir"
    ./build/release/tiledmatmul_test -e data/matmul_tiled/$dir/output.raw -i data/matmul_tiled/$dir/input0.raw,data/matmul_tiled/$dir/input1.raw -t matrix
done

## TILED COARSE MATRIX MULTIPLICATION ##

# Run datagen if data/tiled_coarse_matrix does not exist
if [ ! -d data/matmul_tiled_coarse ]; then
    ./build/release/tcmatmul_datagen
fi

echo
echo "*******************************************"
echo "** Running tests for tiled coarse matmul **"
echo "*******************************************"

# Loop through directories 0 to 8
for dir in $(seq 0 8)
do
    echo "Running test for directory $dir"
    ./build/release/tcmatmul_test -e data/matmul_tiled_coarse/$dir/output.raw -i data/matmul_tiled_coarse/$dir/input0.raw,data/matmul_tiled_coarse/$dir/input1.raw -t matrix
done