#ifndef COAL_RAM_H
#define COAL_RAM_H

__global__ void coalRamMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void coalRamMatrixMul(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

#endif