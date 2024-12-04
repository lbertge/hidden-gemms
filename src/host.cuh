#pragma once

#include <cublas_v2.h>

void native_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void coal_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void shared_memory_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void block_tiling_1d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void block_tiling_2d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void vectorized_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void double_buffered_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void cublas_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle);