#include "kernels.cuh"
#include <cublas_v2.h>
#include <cmath> // For fabs
#include <iostream>
#include <random>

#define TILE_WIDTH 32
#define CEIL(M, N) (((M) + (N) - 1) / (N))

#define EPSILON 1e-1

bool compare_results(const float* kernel, const float* cublas, int M, int N) {
    bool match = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(kernel[i] - cublas[i]) > EPSILON) {
            match = false;
            std::cout << "Mismatch at index " << i 
                      << " (row " << i / N << ", col " << i % N << "): "
                      << "Kernel result = " << kernel[i] 
                      << ", Cublas result = " << cublas[i] 
                      << ", diff = " << std::fabs(kernel[i] - cublas[i]) 
                      << '\n';
        }
    }
    return match;
}

// Naive host
void naive_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    
    naive_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
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

// Tiled Matrix Multiplication 1D host
void block_tiling_1d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    // Block 1d tiling parameters
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    dim3 gridDim(CEIL(M, BM), CEIL(N, BN));
    dim3 blockDim(BM / TM * BN);
    block_tiling_1d_kernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}


// Tiled Matrix Multiplication 2D host
void block_tiling_2d_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    // Block 2d tiling parameters
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 gridDim(CEIL(M, BM), CEIL(N, BN));
    dim3 blockDim(BM / TM * BN / TN);
    block_tiling_2d_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}


// Vectorized host
void vectorized_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    // Block 2d tiling parameters
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 gridDim(CEIL(M, BM), CEIL(N, BN));
    dim3 blockDim(BM / TM * BN / TN);
    vectorized_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}


// Double Buffered host
void double_buffered_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    // Block 2d tiling parameters
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 gridDim(CEIL(M, BM), CEIL(N, BN));
    dim3 blockDim(BM / TM * BN / TN);
    double_buffered_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}


// cuBLAS host
void cublas_host(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}


// Populate matrix
void populate_matrix(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = distribution(generator);
    for (int i = 0; i < K * N; i++) h_B[i] = distribution(generator);
    for (int i = 0; i < M * N; i++) h_C[i] = distribution(generator);
}


// Naive benchmark
void naive_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    naive_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        naive_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// CoalRam benchmark
void coal_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    coal_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        coal_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Shared Memory benchmark
void shared_memory_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    shared_memory_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        shared_memory_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Tiled Matrix Multiplication 1D benchmark
void block_tiling_1d_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    block_tiling_1d_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        block_tiling_1d_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Tiled Matrix Multiplication 2D benchmark
void block_tiling_2d_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    block_tiling_2d_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        block_tiling_2d_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Vectorized benchmark
void vectorized_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    vectorized_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        vectorized_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Double Buffered benchmark
void double_buffered_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Warmup
    double_buffered_host(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        double_buffered_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// cuBLAS benchmark
void cublas_benchmark(float* h_A, float* h_B, float* h_C, float* h_Output, int M, int N, int K, float alpha, float beta, int num_iterations, float* time) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warmup
    cublas_host(d_A, d_B, d_C, M, N, K, alpha, beta, handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        cublas_host(d_A, d_B, d_C, M, N, K, alpha, beta, handle);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaMemcpy(h_Output, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}