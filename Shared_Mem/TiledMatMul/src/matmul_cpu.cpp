#include "matmul_cpu.hpp"

void sgemm_cpu(float *output, float *input0, float *input1,
               int /*numARows*/, int numAColumns, int /*numBRows*/,
               int numBColumns, int numCRows, int numCColumns) {

#define A(i, j) value(input0, i, j, numAColumns)
#define B(i, j) value(input1, i, j, numBColumns)
#define C(i, j) value(output, i, j, numCColumns)
    int ii, jj, kk;
    for (ii = 0; ii < numCRows; ++ii) {
        for (jj = 0; jj < numCColumns; ++jj) {
            float sum = 0;
            for (kk = 0; kk < numAColumns; ++kk) {
                sum += A(ii, kk) * B(kk, jj);
            }
            C(ii, jj) = sum;
        }
    }
#undef A
#undef B
#undef C
}