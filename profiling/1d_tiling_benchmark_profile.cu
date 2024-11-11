#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include <assert.h>
#include <random>

// Tile size for shared memory
#define TILE_WIDTH 32

// CUDA kernel for tiled matrix multiplication matching SGEMM
__global__ void tiledMatrixMulKernel(float* A, float* B, float* C, 
                                    int M, int N, int K,
                                    float alpha, float beta) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // first, figure out which output tile we are working on 
    int cRow = blockIdx.x; 
    int cCol = blockIdx.y; 

    // compute which element in the output tile this thread is responsible for
    int tRow = threadIdx.y; 
    int tCol = threadIdx.x; 

    // the actual row & col that we're accessing in this thread
    int row = cRow * TILE_WIDTH + tRow; 
    int col = cCol * TILE_WIDTH + tCol; 

    float sum = 0.0f; 

    // loop over all the tiles that contribute to this output tile
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // load the correct elements for this tile into shared memory
        // this is for A
        if (row < M && t * TILE_WIDTH + tCol < K) {
            As[tRow][tCol] = A[row * K + t * TILE_WIDTH + tCol];
        } else {
            As[tRow][tCol] = 0.0f; 
        }

        // this is for B
        if (t * TILE_WIDTH + tRow < K && col < N) {
            Bs[tRow][tCol] = B[(t * TILE_WIDTH + tRow) * N + col];
        } else {
            Bs[tRow][tCol] = 0.0f; 
        }

        // wait for all threads to finish loading before proceeding
        __syncthreads(); 

        // compute the dot product for the current tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[tRow][k] * Bs[k][tCol];
        }

        // wait for all threads to finish computing the dot product before proceeding
        __syncthreads(); 
    }

    // write the computed value to the correct position in the output tile
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}

// Updated wrapper function
void tiledMatrixMul(float* A, float* B, float* C, 
                    int M, int N, int K,
                    float alpha, float beta) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // Initialize matrices with random values
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = distribution(generator);
    for (int i = 0; i < K * N; i++) h_B[i] = distribution(generator);
    for (int i = 0; i < M * N; i++) h_C[i] = distribution(generator);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    float alpha = 0.5f;
    float beta = 0.5f;
    
    // Single kernel execution for profiling
    tiledMatrixMul(d_A, d_B, d_C, M, N, K, alpha, beta);
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}