![hidden-gemm-title](./hidden-gemm-title.svg)

# Benchmarks

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
