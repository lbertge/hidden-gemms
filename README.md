![hidden-gemm-title](./hidden-gemm-title.svg)


# How to use 
`make list` to see all possible targets. profiling should only be done on kernels in `profiling/`, to eliminate overhead when reading analysis results. 

Profiling example: `ncu -o 1d_tiling_benchmark_profile --set full bin/1d_tiling_benchmark_profile`

# Link to Google Colab
[Colab](https://colab.research.google.com/drive/15cw6D4iKajNbB4w4P2kj0t1y6SwOJ9dR?usp=sharing)

# FAQ 
* Absolute error tolerance is 0.015, is this high? 
In other matmul implementations I have found, this seems OK, relative error tolerance is much smaller (10e-4).

# Benchmarks

# The following benchmarks run on an RTX 4090
## Vectorized Loads
```
Matrix dimensions: M=1024, N=1024, K=1024
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
Vectorized implementation: 0.125 ms (17147.60 GFLOP/s)
cuBLAS implementation: 0.051 ms (41700.65 GFLOP/s)
Performance ratio (cuBLAS/Vectorized): 2.43x
PASSED

Matrix dimensions: M=2048, N=2048, K=2048
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
Vectorized implementation: 0.372 ms (46230.96 GFLOP/s)
cuBLAS implementation: 0.277 ms (61977.16 GFLOP/s)
Performance ratio (cuBLAS/Vectorized): 1.34x
PASSED

Matrix dimensions: M=4096, N=4096, K=4096
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
Vectorized implementation: 2.934 ms (46839.20 GFLOP/s)
cuBLAS implementation: 2.446 ms (56183.91 GFLOP/s)
Performance ratio (cuBLAS/Vectorized): 1.20x
PASSED

Matrix dimensions: M=8192, N=8192, K=8192
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
Vectorized implementation: 28.723 ms (38280.08 GFLOP/s)
cuBLAS implementation: 19.219 ms (57208.26 GFLOP/s)
Performance ratio (cuBLAS/Vectorized): 1.49x
PASSED
```

## 2D Blocktiling
```
Matrix dimensions: M=1024, N=1024, K=1024
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
2D Tiling implementation: 0.149 ms (14373.90 GFLOP/s)
cuBLAS implementation: 0.051 ms (41775.94 GFLOP/s)
Performance ratio (cuBLAS/2D Tiling): 2.91x
PASSED                                                                                                                                                                "automl-competition" 10:09 05-Dec-24

Matrix dimensions: M=2048, N=2048, K=2048
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
2D Tiling implementation: 0.464 ms (37003.12 GFLOP/s)
cuBLAS implementation: 0.278 ms (61766.80 GFLOP/s)
Performance ratio (cuBLAS/2D Tiling): 1.67x
PASSED

Matrix dimensions: M=4096, N=4096, K=4096
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
2D Tiling implementation: 3.690 ms (37250.62 GFLOP/s)
cuBLAS implementation: 2.437 ms (56389.27 GFLOP/s)
Performance ratio (cuBLAS/2D Tiling): 1.51x
PASSED

Matrix dimensions: M=8192, N=8192, K=8192
Block tiling params: BM=128, BN=128, BK=8, TM=8, TN=8
2D Tiling implementation: 33.185 ms (33133.21 GFLOP/s)
cuBLAS implementation: 19.106 ms (57547.68 GFLOP/s)
Performance ratio (cuBLAS/2D Tiling): 1.74x
```

## Blocktiling 

```
Matrix dimensions: M=1024, N=1024, K=1024
Block tiling params: BM=128, BN=128, BK=2, TM=64
1D Block Tiling: 0.238 ms (9004.52 GFLOP/s)
cuBLAS: 0.051 ms (42024.46 GFLOP/s)
Performance ratio (custom/cuBLAS): 0.21x

Matrix dimensions: M=2048, N=2048, K=2048
Block tiling params: BM=128, BN=128, BK=2, TM=64
1D Block Tiling: 0.690 ms (24888.32 GFLOP/s)
cuBLAS: 0.276 ms (62253.12 GFLOP/s)
Performance ratio (custom/cuBLAS): 0.40x

Matrix dimensions: M=4096, N=4096, K=4096
Block tiling params: BM=128, BN=128, BK=2, TM=64
1D Block Tiling: 5.517 ms (24912.80 GFLOP/s)
cuBLAS: 2.443 ms (56254.55 GFLOP/s)
Performance ratio (custom/cuBLAS): 0.44x

Matrix dimensions: M=8192, N=8192, K=8192
Block tiling params: BM=128, BN=128, BK=2, TM=64
1D Block Tiling: 43.408 ms (25329.42 GFLOP/s)
cuBLAS: 19.282 ms (57024.14 GFLOP/s)
Performance ratio (custom/cuBLAS): 0.44x
```

## Tiling 
```
Matrix dimensions: M=512, N=512, K=512
Tiling implementation: 0.043 ms
cuBLAS implementation: 0.012 ms
Performance ratio (cuBLAS/tiling): 3.66x
Maximum difference from cuBLAS: 2.136230e-04

Matrix dimensions: M=1024, N=1024, K=1024
Tiling implementation: 0.329 ms
cuBLAS implementation: 0.050 ms
Performance ratio (cuBLAS/tiling): 6.54x
Maximum difference from cuBLAS: 5.798340e-04

Matrix dimensions: M=2048, N=2048, K=2048
Tiling implementation: 2.612 ms
cuBLAS implementation: 0.274 ms
Performance ratio (cuBLAS/tiling): 9.52x
Maximum difference from cuBLAS: 0.000000e+00

Matrix dimensions: M=4096, N=4096, K=4096
Tiling implementation: 22.097 ms
cuBLAS implementation: 2.568 ms
Performance ratio (cuBLAS/tiling): 8.60x
Maximum difference from cuBLAS: 0.000000e+00
```
