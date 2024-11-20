#ifndef TILING_H
#define TILING_H

__global__ void tilingMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void tilingMatrixMul(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

#endif