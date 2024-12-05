#include "../src/host.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <chrono>
#include <assert.h>
#include <random>
#include <iostream>

// Benchmark function
void benchmark(int M, int N, int K, int num_iterations = 10) {
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices
    populate_matrix(h_A, h_B, h_C, M, N, K);

    float alpha = 0.5f;
    float beta = 0.5f;

    float *h_Cublas = new float[M * N];
    float cublas_time;
    cublas_benchmark(h_A, h_B, h_C, h_Cublas, M, N, K, alpha, beta, num_iterations, &cublas_time);
    cublas_time /= num_iterations;

    // Calculate GFLOPS
    double operations = 2.0 * M * N * K;  // multiply-adds
    double cublas_gflops = (operations * 1e-9) / (cublas_time * 1e-3);

    // Print results
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("cuBLAS implementation: %.3f ms (%.2f GFLOP/s)\n", cublas_time, cublas_gflops);

    float *h_Naive = new float[M * N];
    float naive_time;
    naive_benchmark(h_A, h_B, h_C, h_Naive, M, N, K, alpha, beta, num_iterations, &naive_time);
    naive_time /= num_iterations;

    // Calculate GFLOPS
    double naive_gflops = (operations * 1e-9) / (naive_time * 1e-3);

    // Print results
    printf("Naive implementation: %.3f ms (%.2f GFLOP/s)\n", naive_time, naive_gflops);

    // Compare results
    if (!compare_results(h_Naive, h_Cublas, M, N)) {
        printf("Naive and cuBLAS results do not match!\n");
    }

    delete[] h_Naive;

    float *h_Coal = new float[M * N];
    float coal_time;
    coal_benchmark(h_A, h_B, h_C, h_Coal, M, N, K, alpha, beta, num_iterations, &coal_time);
    coal_time /= num_iterations;

    // Calculate GFLOPS
    double coal_gflops = (operations * 1e-9) / (coal_time * 1e-3);

    // Print results
    printf("Coalesced implementation: %.3f ms (%.2f GFLOP/s)\n", coal_time, coal_gflops);

    // Compare results
    if (!compare_results(h_Coal, h_Cublas, M, N)) {
        printf("Coalesced and cuBLAS results do not match!\n");
    }

    delete[] h_Coal;

    float *h_Shared = new float[M * N];
    float shared_time;
    shared_memory_benchmark(h_A, h_B, h_C, h_Shared, M, N, K, alpha, beta, num_iterations, &shared_time);
    shared_time /= num_iterations;

    // Calculate GFLOPS
    double shared_gflops = (operations * 1e-9) / (shared_time * 1e-3);

    // Print results
    printf("Shared memory implementation: %.3f ms (%.2f GFLOP/s)\n", shared_time, shared_gflops);

    // Compare results
    if (!compare_results(h_Shared, h_Cublas, M, N)) {
        printf("Shared memory and cuBLAS results do not match!\n");
    }

    delete[] h_Shared;

    float *h_Block1D = new float[M * N];
    float block1D_time;
    block_tiling_1d_benchmark(h_A, h_B, h_C, h_Block1D, M, N, K, alpha, beta, num_iterations, &block1D_time);
    block1D_time /= num_iterations;

    // Calculate GFLOPS
    double block1D_gflops = (operations * 1e-9) / (block1D_time * 1e-3);

    // Print results
    printf("Block tiling 1D implementation: %.3f ms (%.2f GFLOP/s)\n", block1D_time, block1D_gflops);

    // Compare results
    if (!compare_results(h_Block1D, h_Cublas, M, N)) {
        printf("Block tiling 1D and cuBLAS results do not match!\n");
    }

    delete[] h_Block1D;

    float *h_Block2D = new float[M * N];
    float block2D_time;
    block_tiling_2d_benchmark(h_A, h_B, h_C, h_Block2D, M, N, K, alpha, beta, num_iterations, &block2D_time);
    block2D_time /= num_iterations;

    // Calculate GFLOPS
    double block2D_gflops = (operations * 1e-9) / (block2D_time * 1e-3);

    // Print results
    printf("Block tiling 2D implementation: %.3f ms (%.2f GFLOP/s)\n", block2D_time, block2D_gflops);

    // Compare results
    if (!compare_results(h_Block2D, h_Cublas, M, N)) {
        printf("Block tiling 2D and cuBLAS results do not match!\n");
    }

    delete[] h_Block2D;

    float *h_Vectorized = new float[M * N];
    float vectorized_time;
    vectorized_benchmark(h_A, h_B, h_C, h_Vectorized, M, N, K, alpha, beta, num_iterations, &vectorized_time);
    vectorized_time /= num_iterations;

    // Calculate GFLOPS
    double vectorized_gflops = (operations * 1e-9) / (vectorized_time * 1e-3);

    // Print results
    printf("Vectorized implementation: %.3f ms (%.2f GFLOP/s)\n", vectorized_time, vectorized_gflops);

    // Compare results
    if (!compare_results(h_Vectorized, h_Cublas, M, N)) {
        printf("Vectorized and cuBLAS results do not match!\n");
    }

    delete[] h_Vectorized;

    // float *h_DoubleBuffered = new float[M * N];
    // float double_buffered_time;
    // double_buffered_benchmark(h_A, h_B, h_C, h_DoubleBuffered, M, N, K, alpha, beta, num_iterations, &double_buffered_time);
    // double_buffered_time /= num_iterations;

    // // Calculate GFLOPS
    // double double_buffered_gflops = (operations * 1e-9) / (double_buffered_time * 1e-3);

    // // Print results
    // printf("Double buffered implementation: %.3f ms (%.2f GFLOP/s)\n", double_buffered_time, double_buffered_gflops);

    // // Compare results
    // if (!compare_results(h_DoubleBuffered, h_Cublas, M, N)) {
    //     printf("Double buffered and cuBLAS results do not match!\n");
    // }

    // delete[] h_DoubleBuffered;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_Cublas;

    // Print performance ratios
    printf("Performance ratios:\n");
    printf("  cuBLAS/naive: %.2fx\n", naive_time / cublas_time);
    printf("  cuBLAS/coalesced: %.2fx\n", coal_time / cublas_time);
    printf("  cuBLAS/shared: %.2fx\n", shared_time / cublas_time);
    printf("  cuBLAS/block1D: %.2fx\n", block1D_time / cublas_time);
    printf("  cuBLAS/block2D: %.2fx\n", block2D_time / cublas_time);
    printf("  cuBLAS/vectorized: %.2fx\n", vectorized_time / cublas_time);
    // printf("  cuBLAS/double_buffered: %.2fx\n", double_buffered_time / cublas_time);
}

int main() {
    for (int size = 256; size <= 8192; size += 256) {
        benchmark(size, size, size);
        printf("\n");
    }
    
    return 0;
} 