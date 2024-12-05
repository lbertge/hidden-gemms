#pragma once

#define TILE_WIDTH 32

// Shared Memory Kernel Implementation
__global__ void shared_memory_kernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    int AStart = by * TILE_WIDTH * K;
    int BStart = bx * TILE_WIDTH;
    int AStep = TILE_WIDTH;
    int BStep = TILE_WIDTH * N;
    int AEnd = AStart + K - 1;

    float sum = 0.0f;

    for (int a = AStart, b = BStart; a <= AEnd; a += AStep, b += BStep) {

        As[ty][tx] = A[a + K * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        __syncthreads(); 

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); 
    }

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    int idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}