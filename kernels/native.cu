#define TILE_WIDTH 32
#define CEIL(M, N) (((M) + (N) - 1) / (N))

// Naive kernel implementation
__global__ void naiveMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Naive wrapper function
void naiveMatrixMul(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 gridDim(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    
    naiveMatrixMulKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}