#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>

// Tile size for shared memory
#define TILE_WIDTH 32

// CUDA kernel for tiled matrix multiplication
__global__ void tiledMatrixMulKernel(float* A, float* B, float* C, 
                                    int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate row and column for this thread
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_WIDTH + tx < K)
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t * TILE_WIDTH + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Wrapper function for kernel launch
void tiledMatrixMul(float* A, float* B, float* C, int M, int N, int K) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
}

// Benchmark function
void benchmark(int M, int N, int K, int num_iterations = 10) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_cublas = (float*)malloc(size_C);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_cublas;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_cublas, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Constants for cublasSgemm
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warmup
    tiledMatrixMul(d_A, d_B, d_C, M, N, K);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N, d_A, K, &beta,
                d_C_cublas, N);
    
    // Benchmark custom implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        tiledMatrixMul(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    custom_time /= num_iterations;
    
    // Benchmark cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K, &beta,
                    d_C_cublas, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublas_time /= num_iterations;
    
    // Print results
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tiling implementation: %.3f ms\n", custom_time);
    printf("cuBLAS implementation: %.3f ms\n", cublas_time);
    printf("Performance ratio (cuBLAS/tiling): %.2fx\n", custom_time/cublas_time);
    
    // Verify results
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C[i] - h_C_cublas[i]);
        max_diff = max(max_diff, diff);
    }
    printf("Maximum difference from cuBLAS: %e\n", max_diff);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cublas);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with different matrix sizes
    int sizes[] = {512, 1024, 2048, 4096};
    
    for (int size : sizes) {
        benchmark(size, size, size);
        printf("\n");
    }
    
    return 0;
}