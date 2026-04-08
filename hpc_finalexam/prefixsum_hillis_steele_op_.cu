#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

// Macro for robust error checking [7]
#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t result = expr_to_check; \
    if(result != cudaSuccess) { \
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Optimized Prefix Sum using Double Buffering
__global__ void prefixSumDoubleBuffered(int *a, int *b, int n) {
    // 1. Double-buffered shared memory to eliminate read-after-write hazards
    __shared__ int temp[2][BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    
    int pout = 0;
    int pin = 1;
    
    // Load data into the initial shared memory buffer
    if (gid < n) {
        temp[pout][tid] = a[gid];
    } else {
        temp[pout][tid] = 0;
    }
    __syncthreads(); // Wait for all threads to finish loading [1]
    
    // 2. Perform the scan with halved synchronization overhead
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout; // Swap the write buffer
        pin  = 1 - pout; // Swap the read buffer
        
        if (tid >= offset) {
            // Read from the 'pin' buffer and write to the 'pout' buffer
            temp[pout][tid] = temp[pin][tid] + temp[pin][tid - offset];
        } else {
            // Just pass the value forward if it has no offset pair
            temp[pout][tid] = temp[pin][tid];
        }
        
        // Only one synchronization barrier is needed per iteration now [1]
        __syncthreads(); 
    }

    // Write the final computed result back to global memory
    if (gid < n) {
        b[gid] = temp[pout][tid];
    }
}

int main() {
    int *a, *b;
    int n = 1024;
    size_t size = n * sizeof(int);
    
    // 3. Allocate page-locked (pinned) memory for faster transfers [4, 5]
    CUDA_CHECK(cudaMallocHost(&a, size));
    CUDA_CHECK(cudaMallocHost(&b, size));
    
    for(int i = 0; i < n; i++) {
        a[i] = i + 1;
    }

    int *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    
    // Transfers are substantially faster because 'a' is page-locked [5]
    CUDA_CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = BLOCK_SIZE;
    // 4. Ceiling division to safely support arbitrary array sizes 'n' [6]
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; 
    
    prefixSumDoubleBuffered<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < n; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
    
    // Free page-locked memory [5]
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}