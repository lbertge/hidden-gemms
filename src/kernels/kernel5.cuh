#pragma once

// 2D block tiling kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void block_tiling_2d_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x % (BN / TN) * TN;
    int ty = threadIdx.x / (BN / TN) * TM;

    int AStart = by * BM * K;
    int BStart = bx * BN;
    int CStart = by * BM * N + bx * BN;

    int AsRow = threadIdx.x / BK;
    int AsCol = threadIdx.x % BK;

    int BsRow = threadIdx.x / BN;
    int BsCol = threadIdx.x % BN;

    float reg[TM][TN] = {0.0f};

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        As[AsRow][AsCol] = A[AStart + AsRow * K + AsCol + k];
        Bs[BsRow][BsCol] = B[BStart + BsRow * N + BsCol + k * N];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                for (int l = 0; l < TN; ++l) {
                    reg[j][l] += As[ty + j][i] * Bs[i][tx + l];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[CStart + (ty + i) * N + tx + j] = alpha * reg[i][j] + beta * C[CStart + (ty + i) * N + tx + j];
        }
    }
}