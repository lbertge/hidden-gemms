#include <cuda_runtime.h>
#include <stdio.h>

void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max block dimensions: %dx%dx%d\n", 
            prop.maxThreadsDim[0], 
            prop.maxThreadsDim[1], 
            prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: %dx%dx%d\n", 
            prop.maxGridSize[0], 
            prop.maxGridSize[1], 
            prop.maxGridSize[2]);
        printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate * 1e-6);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache size: %d KB\n\n", prop.l2CacheSize / 1024);
    }
}

int main() {
    printDeviceProperties();
    return 0;
} 