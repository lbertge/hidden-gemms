#pragma once

template <const int BM, const int BN, const int BK, const int TM>
__global__ void block_tiling_1d_kernel(const float *A, const float *B, float *C,
                                   int M, int N, int K, float alpha, float beta
                                   ) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x;
  const int threadRow = threadIdx.y;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // Move blocktile to beginning of A's row and B's column
  // A += cRow * BM * K;
  // B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  // assert(BM * BK == blockDim.x);
  // assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = (threadIdx.y * TM) + (threadIdx.x / BK);
  const uint innerColB = threadIdx.x; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.y;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA][innerColA] = A[cRow * BM * K + bkIdx + innerRowA * K + innerColA];
    Bs[innerRowB][innerColB] = B[cCol * BN + bkIdx * N + innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    // A += BK;
    // B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx][threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[threadRow * TM + resIdx][dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}