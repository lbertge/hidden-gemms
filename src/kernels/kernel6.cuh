#pragma once

// Vectorize kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void vectorized_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = (threadIdx.x % (BN / TN)) * TN;
    const int ty = (threadIdx.x / (BN / TN)) * TM;
    const int thread_num = BM / TM * BN / TN;

    const int AStart = by * BM * K;
    const int BStart = bx * BN;
    const int CStart = by * BM * N + bx * BN;

    const int As_vec_num = BM * BK / thread_num / 4;
    const int Bs_vec_num = BK * BN / thread_num / 4;

    int AsRow = threadIdx.x / (BK / 4);
    int AsCol = threadIdx.x % (BK / 4) * 4;
    int AsStep = BM / As_vec_num;

    int BsRow = threadIdx.x / (BN / 4);
    int BsCol = threadIdx.x % (BN / 4) * 4;
    int BsStep = BK / Bs_vec_num;

    float sum[TM][TN] = {0.0f};
    float tmp_a[TM] = {0.0f};
    float tmp_b[TN] = {0.0f};

    float A_trans_temp[4 * As_vec_num] = {0.0f};

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += AsStep) {
            int vec_num = i / AsStep * 4;
            reinterpret_cast<float4 *>(&A_trans_temp[vec_num])[0] = reinterpret_cast<const float4 *>(&A[AStart + (AsRow + i) * K + AsCol + k])[0];
            As[AsCol][AsRow + i] = A_trans_temp[vec_num];
            As[AsCol + 1][AsRow + i] = A_trans_temp[vec_num + 1];
            As[AsCol + 2][AsRow + i] = A_trans_temp[vec_num + 2];
            As[AsCol + 3][AsRow + i] = A_trans_temp[vec_num + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += BsStep) {
            reinterpret_cast<float4 *>(&Bs[BsRow + i][BsCol])[0] = reinterpret_cast<const float4 *>(&B[BStart + (BsRow + i) * N + BsCol + k * N])[0];
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; j += 4) {
                reinterpret_cast<float4 *>(&tmp_a[j])[0] = reinterpret_cast<const float4 *>(&As[i][ty + j])[0];
            }

            #pragma unroll
            for (int l = 0; l < TN; l += 4) {
                reinterpret_cast<float4 *>(&tmp_b[l])[0] = reinterpret_cast<const float4 *>(&Bs[i][tx + l])[0];
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
            float4 tmp_c = reinterpret_cast<float4 *>(&C[CStart + (ty + i) * N + tx + j])[0];
            tmp_c.x = alpha * sum[i][j] + beta * tmp_c.x;
            tmp_c.y = alpha * sum[i][j + 1] + beta * tmp_c.y;
            tmp_c.z = alpha * sum[i][j + 2] + beta * tmp_c.z;
            tmp_c.w = alpha * sum[i][j + 3] + beta * tmp_c.w;
            reinterpret_cast<float4 *>(&C[CStart + (ty + i) * N + tx + j])[0] = tmp_c;
        }
    }
}