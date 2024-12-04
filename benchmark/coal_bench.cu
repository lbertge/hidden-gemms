#include "../src/host.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include <assert.h>
#include <random>

// Benchmark function
void benchmark(int M, int N, int K, int num_iterations = 10) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_cublas = new float[M * N];
    
    // Initialize matrices
    int some_seed = 759;
    std::mt19937 generator(some_seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < M * K; i++) h_A[i] = distribution(generator);
    for (int i = 0; i < K * N; i++) h_B[i] = distribution(generator);
    for (int i = 0; i < M * N; i++) h_C[i] = distribution(generator);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_cublas;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_cublas, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_cublas, h_C, size_C, cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Constants for SGEMM
    float alpha = 0.5f;
    float beta = 0.5f;
    
    // Warmup
    coal_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    cublas_host(d_A, d_B, d_C_cublas, M, N, K, alpha, beta, handle);
    
    // Benchmark naive implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float coal_ram_time;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        coal_host(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&coal_ram_time, start, stop);
    coal_ram_time /= num_iterations;
    
    // Benchmark cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        cublas_host(d_A, d_B, d_C_cublas, M, N, K, alpha, beta, handle);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublas_time /= num_iterations;
    
    // Calculate GFLOPS
    double operations = 2.0 * M * N * K;  // multiply-adds
    double naive_gflops = (operations * 1e-9) / (coal_ram_time * 1e-3);
    double cublas_gflops = (operations * 1e-9) / (cublas_time * 1e-3);
    
    // Print results
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Coal Ram implementation: %.3f ms (%.2f GFLOP/s)\n", coal_ram_time, naive_gflops);
    printf("cuBLAS implementation: %.3f ms (%.2f GFLOP/s)\n", cublas_time, cublas_gflops);
    printf("Performance ratio:\n");
    printf("  cuBLAS/coal_ram: %.2fx\n", coal_ram_time/cublas_time);
    
    // Verify results
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);
    
    // Check results
    float epsilon = 1e-2;
    for (int i = 0; i < M * N; i++) {
        assert(fabs(h_C[i] - h_C_cublas[i]) < epsilon);
    }
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cublas;
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