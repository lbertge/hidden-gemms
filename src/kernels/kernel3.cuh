#define TILE_WIDTH 32

// Shared Memory Kernel Implementation
__global__ void shared_memory_kernel(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // first, figure out which output tile we are working on 
    int cRow = blockIdx.x; 
    int cCol = blockIdx.y; 

    // compute which element in the output tile this thread is responsible for
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    // the actual row & col that we're accessing in this thread
    int AStart = cRow * TILE_WIDTH * K;
    int BStart = cCol * TILE_WIDTH;
    int AStep = TILE_WIDTH;
    int BStep = TILE_WIDTH * N;
    int AEnd = AStart + K - 1;

    float sum = 0.0f;

    for (int a = AStart, b = BStart; a <= AEnd; a += AStep, b += BStep) {

        As[ty][tx] = A[a + K * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // wait for all threads to finish loading before proceeding
        __syncthreads(); 

        // compute the dot product for the current tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // wait for all threads to finish computing the dot product before proceeding
        __syncthreads(); 
    }

    int idx = N * TILE_WIDTH * cRow + TILE_WIDTH * cCol;
    C[idx + N * ty + tx] = alpha * sum + beta * C[idx + N * ty + tx];
}