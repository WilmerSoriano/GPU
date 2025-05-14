#pragma once

#define value(arry, i, j, width) arry[(i) * width + (j)]

void sgemm_cpu(float *output, float *input0, float *input1,
               int /*numARows*/, int numAColumns, int /*numBRows*/,
               int numBColumns, int numCRows, int numCColumns);