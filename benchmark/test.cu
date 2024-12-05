#include "../src/host.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include <assert.h>
#include <random>
#include <iostream>

// Block tiling parameters
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

// Benchmark function
void benchmark(int M, int N, int K, int num_iterations = 10) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_kernel = new float[M * N];
    
    // Initialize matrices
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = distribution(generator);
    for (int i = 0; i < K * N; i++) h_B[i] = distribution(generator);
    for (int i = 0; i < M * N; i++) h_C[i] = distribution(generator);  // Initialize C as well
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    // Constants for SGEMM
    float alpha = 0.5f;
    float beta = 0.5f;
    
    // Warmup runs
    double_buffered_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    
    // Benchmark custom implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        double_buffered_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    custom_time /= num_iterations;
    
    // Calculate GFLOPS
    double operations = 2.0 * M * N * K;
    double custom_gflops = (operations * 1e-9) / (custom_time * 1e-3);
    
    // Print results
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block tiling params: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
    printf("Double Buffering implementation: %.3f ms (%.2f GFLOP/s)\n", custom_time, custom_gflops);
    
    // Verify results
    cudaMemcpy(h_C_kernel, d_C, size_C, cudaMemcpyDeviceToHost);

    cpu_gemm(h_A, h_B, h_C, M, N, K, alpha, beta);

    if (compare_results(h_C, h_C_kernel, M, N)) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED\n";
    }
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_kernel;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with different matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048};
    
    for (int size : sizes) {
        benchmark(size, size, size);
        printf("\n");
    }
    
    return 0;
} 