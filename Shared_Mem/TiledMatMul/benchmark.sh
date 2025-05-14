#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

module load cuda/12.2

make benchmark

ncu --set full -o big_benchmark -f ./build/main_benchmark 1024 1024 1024

module unload cuda/12.2