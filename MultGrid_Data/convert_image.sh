#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

./build/release/rgb2gray images/gpgpu.png
