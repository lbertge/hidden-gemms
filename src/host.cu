#include "kernels/kernel1.cuh"
#include "kernels/kernel2.cuh"
#include "kernels/kernel3.cuh"
#include "kernels/kernel4.cuh"
#include "kernels/kernel5.cuh"
#include "kernels/kernel6.cuh"

#define TILE_WIDTH 32
#define CEIL(M, N) (((M) + (N) - 1) / (N))

// Naive host
void native_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    
    native_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

// CoalRam host
void coal_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH * TILE_WIDTH);
    
    coal_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

// Shared Memory host
void shared_memory_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    
    shared_memory_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// Block tiling parameters
const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;

// Tiled Matrix Multiplication 1D host
void block_tiling_1d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(N, BN), CEIL(M, BM));
    dim3 blockDim(BN, BM / TM);
    block_tiling_1d_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}