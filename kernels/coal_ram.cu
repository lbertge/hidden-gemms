// Tile size for shared memory
#define TILE_WIDTH 32
#define CEIL(M, N) (((M) + (N) - 1) / (N))

// Naive kernel implementation
__global__ void coalRamMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
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

// Naive wrapper function
void coalRamMatrixMul(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    
    coalRamMatrixMulKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}