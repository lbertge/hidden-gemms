#ifndef NATIVE_H
#define NATIVE_H

__global__ void naiveMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void naiveMatrixMul(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

#endif