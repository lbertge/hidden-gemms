#pragma once

void native_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void coal_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void shared_memory_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);

void block_tiling_1d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta);