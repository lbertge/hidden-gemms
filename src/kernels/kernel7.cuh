#pragma once

#define vec(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// Double Buffering Kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void double_buffered_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    // As is BK x BM because we transpose As
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = (threadIdx.x % (BN / TN)) * TN;
    int ty = (threadIdx.x / (BN / TN)) * TM;
    const int thread_num = BM / TM * BN / TN;

    int AStart = by * BM * K;
    int BStart = bx * BN;
    int CStart = by * BM * N + bx * BN;

    // How many vectors we need to load in a thread
    const int As_vec_num = BM * BK / thread_num / 4;
    const int Bs_vec_num = BK * BN / thread_num / 4;

    int AsRow = threadIdx.x / (BK / 4);
    // float4 is along the row so we need to multiply by 4 for column
    int AsCol = threadIdx.x % (BK / 4) * 4;
    // AsStep is the step size for each load given a thread
    int AsStep = BM / As_vec_num;

    int BsRow = threadIdx.x / (BN / 4);
    int BsCol = threadIdx.x % (BN / 4) * 4;
    int BsStep = BK / Bs_vec_num;

    float sum[TM][TN] = {0.0f};
    float tmp_a[TM];
    float tmp_b[TN];

    float A_trans_temp[4 * As_vec_num];

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += AsStep) {
            // We need to do the transpose so we can use vectorization later
            int vec_num = i / AsStep * 4;
            vec(A_trans_temp[vec_num]) = vec(A[AStart + (AsRow + i) * K + AsCol + k]);
            As[AsCol][AsRow + i] = A_trans_temp[vec_num];
            As[AsCol + 1][AsRow + i] = A_trans_temp[vec_num + 1];
            As[AsCol + 2][AsRow + i] = A_trans_temp[vec_num + 2];
            As[AsCol + 3][AsRow + i] = A_trans_temp[vec_num + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += BsStep) {
            vec(Bs[BsRow + i][BsCol]) = vec(B[BStart + (BsRow + i) * N + BsCol + k * N]);
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; j += 4) {
                vec(tmp_a[j]) = vec(As[i][ty + j]);
            }

            #pragma unroll
            for (int l = 0; l < TN; l += 4) {
                vec(tmp_b[l]) = vec(Bs[i][tx + l]);
            }

            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                for (int l = 0; l < TN; ++l) {
                    sum[j][l] += tmp_a[j] * tmp_b[l];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            float4 tmp_c = vec(C[CStart + (ty + i) * N + tx + j]);
            tmp_c.x = alpha * sum[i][j] + beta * tmp_c.x;
            tmp_c.y = alpha * sum[i][j + 1] + beta * tmp_c.y;
            tmp_c.z = alpha * sum[i][j + 2] + beta * tmp_c.z;
            tmp_c.w = alpha * sum[i][j + 3] + beta * tmp_c.w;
            vec(C[CStart + (ty + i) * N + tx + j]) = tmp_c;
        }
    }
}