#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 = device ID

    printf("GPU Name: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);

    return 0;
}