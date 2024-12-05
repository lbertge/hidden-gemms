#pragma once

#define vec(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

// Double Buffering
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void double_buffered_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = (threadIdx.x % (BN / TN)) * TN;
    int ty = (threadIdx.x / (BN / TN)) * TM;
    const int thread_num = BM / TM * BN / TN;

    int AStart = by * BM * K;
    int BStart = bx * BN;
    int CStart = by * BM * N + bx * BN;

    const int As_vec_num = BM * BK / thread_num / 4;
    const int Bs_vec_num = BK * BN / thread_num / 4;

    int AsRow = threadIdx.x / (BK / 4);
    int AsCol = threadIdx.x % (BK / 4) * 4;
    int AsStep = BM / As_vec_num;

    int BsRow = threadIdx.x / (BN / 4);
    int BsCol = threadIdx.x % (BN / 4) * 4;
    int BsStep = BK / Bs_vec_num;

    float sum[TM][TN] = {0.0f};
    float tmp_a[2][TM];
    float tmp_b[2][TN];

    float A_trans_temp[4 * As_vec_num];

    // Double Buffering Loop initialization
    #pragma unroll
    for (int i = 0; i < BM; i += AsStep) {
        int vec_num = i / AsStep * 4;
        vec(A_trans_temp[vec_num]) = vec(A[AStart + (AsRow + i) * K + AsCol + k]);
        As[0][AsCol][AsRow + i] = A_trans_temp[vec_num];
        As[0][AsCol + 1][AsRow + i] = A_trans_temp[vec_num + 1];
        As[0][AsCol + 2][AsRow + i] = A_trans_temp[vec_num + 2];
        As[0][AsCol + 3][AsRow + i] = A_trans_temp[vec_num + 3];
    }
    #pragma unroll
    for (int i = 0; i < BK; i += BsStep) {
        vec(Bs[0][BsRow + i][BsCol]) = vec(B[BStart + (BsRow + i) * N + BsCol + k * N]);
    }

    int k = 0;
    int state = 0;
    do {
        k += BK;
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += AsStep) {
                int vec_num = i / AsStep * 4;
                vec(A_trans_temp[vec_num]) = vec(A[AStart + (AsRow + i) * K + AsCol + k]);
                As[^state][AsCol][AsRow + i] = A_trans_temp[vec_num];
                As[^state][AsCol + 1][AsRow + i] = A_trans_temp[vec_num + 1];
                As[^state][AsCol + 2][AsRow + i] = A_trans_temp[vec_num + 2];
                As[^state][AsCol + 3][AsRow + i] = A_trans_temp[vec_num + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += BsStep) {
                vec(Bs[^state][BsRow + i][BsCol]) = vec(B[BStart + (BsRow + i) * N + BsCol + k * N]);
            }
        }

        __syncthreads();

        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BK; ++i) {
                #pragma unroll
                for (int j = 0; j < TM; j += 4) {
                    vec(tmp_a[state][j]) = vec(As[state][i][ty + j]);
                }

                #pragma unroll
                for (int l = 0; l < TN; l += 4) {
                    vec(tmp_b[state][l]) = vec(Bs[state][i][tx + l]);
                }

                #pragma unroll
                for (int j = 0; j < TM; ++j) {
                    for (int l = 0; l < TN; ++l) {
                        sum[j][l] += tmp_a[state][j] * tmp_b[state][l];
                    }
                }
            }
        }
        state ^= 1;
    } while (k < K);

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