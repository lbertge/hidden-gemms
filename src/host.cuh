#pragma once

#include <cublas_v2.h>

bool compare_results(const float* kernel, const float* cublas, int M, int N);

void naive_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void coal_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void shared_memory_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void block_tiling_1d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void block_tiling_2d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void vectorized_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void double_buffered_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void cublas_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle);

void populate_matrix(float* h_A, float* h_B, float* h_c, int M, int N, int K);

void naive_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void coal_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void shared_memory_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void block_tiling_1d_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void block_tiling_2d_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void vectorized_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void double_buffered_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);

void cublas_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time);