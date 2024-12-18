#pragma once

// 2D block tiling kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void block_tiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = (threadIdx.x % (BN / TN)) * TN;
    int ty = (threadIdx.x / (BN / TN)) * TM;
    int thread_num = blockDim.x;

    int AStart = by * BM * K;
    int BStart = bx * BN;
    int CStart = by * BM * N + bx * BN;

    int AsRow = threadIdx.x / BK;
    int AsCol = threadIdx.x % BK;
    int AsStep = thread_num / BK;

    int BsRow = threadIdx.x / BN;
    int BsCol = threadIdx.x % BN;
    int BsStep = thread_num / BN;

    float reg[TM][TN] = {0.0f};

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // We do need to iterate through since this is 2D block tiling
        #pragma unroll
        for (int i = 0; i < BM; i += AsStep) {
            As[AsRow + i][AsCol] = A[AStart + (AsRow + i) * K + AsCol + k];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += BsStep) {
            Bs[BsRow + i][BsCol] = B[BStart + (BsRow + i) * N + BsCol + k * N];
        }
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