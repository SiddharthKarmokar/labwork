#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: each thread writes its index into the array
__global__ void myKernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] + idx;  // simple operation
}

int main() {
    int N = 1024;
    int size = N * sizeof(float);
    float *data; // ONE pointer to rule them all

    // 1. Single Allocation (Unified Memory)
    cudaMallocManaged(&data, size);

    // Initialize data on CPU
    for (int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    // 2. CPU does work directly
    data[0] = 5.0f;

    // 3. GPU does work directly (No Memcpy!)
    myKernel<<<1, 1024>>>(data);

    // 4. Wait for GPU to finish
    cudaDeviceSynchronize();

    // 5. CPU can access results directly
    printf("First 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }

    // 6. Free memory
    cudaFree(data);

    return 0;
}