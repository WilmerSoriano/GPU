#!/bin/bash

# Test Conv2D
echo "************************"
echo "* Running Conv2D tests *"
echo "************************"

# Loop through directories 0 to 20
for dir in $(seq 0 20)
do
    echo "Running test for directory $dir"
    ./build/release/conv2d_test -e data/$dir/output.raw -i data/$dir/input.raw,data/$dir/kernel.raw -t matrix
done

# Test Conv2d with Constant Memory
echo ""
echo "****************************"
echo "* Running Conv2DCMem tests *"
echo "****************************"

# Loop through directories 0 to 20
for dir in $(seq 0 20)
do
    echo "Running test for directory $dir"
    ./build/release/conv2d_cmem_test -e data/$dir/output.raw -i data/$dir/input.raw,data/$dir/kernel.raw -t matrix
done

# Test Conv2D Tiled
echo ""
echo "************************"
echo "* Running Conv2DTiled tests *"
echo "************************"

# Loop through directories 0 to 20
for dir in $(seq 0 20)
do
    echo "Running test for directory $dir"
    ./build/release/conv2d_tiled_test -e data/$dir/output.raw -i data/$dir/input.raw,data/$dir/kernel.raw -t matrix
done

# Test Conv2D Tiled Cache
echo ""
echo "**********************************"
echo "* Running Conv2DTiledCache tests *"
echo "**********************************"

# Loop through directories 0 to 20
for dir in $(seq 0 20)
do
    echo "Running test for directory $dir"
    ./build/release/conv2d_tiled_cache_test -e data/$dir/output.raw -i data/$dir/input.raw,data/$dir/kernel.raw -t matrix
done