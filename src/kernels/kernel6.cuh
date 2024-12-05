#pragma once

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void vectorized_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    extern __shared__ float shared_mem[]; // Shared memory allocation
    float4 *As = reinterpret_cast<float4*>(shared_mem); // Shared memory for A
    float4 *Bs = reinterpret_cast<float4*>(shared_mem + BM * BK); // Shared memory for B

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int thread_num = blockDim.x;

    // Compute thread coordinates
    int tx = (threadIdx.x % (BN / TN)) * TN; // Thread's column start
    int ty = (threadIdx.x / (BN / TN)) * TM; // Thread's row start
    int thread_id = threadIdx.x;

    int AStart = by * BM * K;
    int BStart = bx * BN;
    int CStart = by * BM * N + bx * BN;

    int AsRow = thread_id / BK; // Row in shared memory for A
    int AsCol = (thread_id % BK) / 4; // Column in shared memory for A (float4 access)
    int BsRow = thread_id / BN; // Row in shared memory for B
    int BsCol = (thread_id % BN) / 4; // Column in shared memory for B (float4 access)

    float4 reg[TM][TN / 4] = {0.0f}; // Registers for results, vectorized by float4

    for (int k = 0; k < K; k += BK) {
        // Load tiles into shared memory
        for (int i = 0; i < BM; i += thread_num / BK) {
            if (AsRow + i < BM && AsCol < BK / 4) {
                As[AsRow + i + AsCol * (BM / 4)] = 
                    reinterpret_cast<const float4*>(A)[(AStart / 4) + (AsRow + i) * (K / 4) + AsCol];
            }
        }

        for (int i = 0; i < BN; i += thread_num / BK) {
            if (BsRow + i < BK && BsCol < BN / 4) {
                Bs[BsRow + i + BsCol * (BK / 4)] = 
                    reinterpret_cast<const float4*>(B)[(BStart / 4) + (BsRow + i) * (N / 4) + BsCol];
            }
        }
        __syncthreads();

        // Compute tile results
        for (int i = 0; i < BK; ++i) {
            float4 vecA[TM];
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                vecA[j] = As[(ty + j) * (BK / 4) + i];
            }

            #pragma unroll
            for (int j = 0; j < TN / 4; ++j) {
                float4 vecB = Bs[i * (BN / 4) + tx / 4 + j];
                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    reg[m][j].x += vecA[m].x * vecB.x;
                    reg[m][j].y += vecA[m].y * vecB.y;
                    reg[m][j].z += vecA[m].z * vecB.z;
                    reg[m][j].w += vecA[m].w * vecB.w;
                }
            }
        }
        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN / 4; ++j) {
            float4 result = reg[i][j];
            int C_index = CStart / 4 + (ty + i) * (N / 4) + tx / 4 + j;
            reinterpret_cast<float4*>(C)[C_index] = result;
        }
    }
}

// Vectorize kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void vectorized_kernel_bk(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
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