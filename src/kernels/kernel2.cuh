#pragma once

#define TILE_WIDTH 32

// CoalRam kernel implementation
__global__ void coal_kernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    const int row = blockIdx.x * TILE_WIDTH + (threadIdx.x / TILE_WIDTH);
    const int col = blockIdx.y * TILE_WIDTH + (threadIdx.x % TILE_WIDTH);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}