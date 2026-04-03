#include <stdio.h>
#include<cuda_runtime.h>

// Define the number of threads per block
#define BLOCK_SIZE 256 

__global__ void prefixSumInclusive(const float *idata, float *odata, int n) {
    // 1. Allocate shared memory for the block
    // We use shared memory because threads will be reading each other's intermediate results
    __shared__ float temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Load data from global memory into shared memory
    // If the global ID is out of bounds, pad with 0.0f so it doesn't affect the sum
    if (gid < n) {
        temp[tid] = idata[gid];
    } else {
        temp[tid] = 0.0f;
    }
    
    // Wait for all threads to finish loading their element
    __syncthreads(); 

    // 3. The Hillis-Steele Scan Loop
    // We increase the look-back offset by powers of 2 (1, 2, 4, 8, 16...)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        
        float val = 0.0f;
        
        // Read the value from the "left" (if the thread index is past the offset)
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        
        // SYNC 1: Ensure all threads have READ their required values before anyone writes.
        // This prevents Read-After-Write (RAW) hazards.
        __syncthreads(); 

        // Add the value to our current position
        if (tid >= offset) {
            temp[tid] += val;
        }
        
        // SYNC 2: Ensure all threads have WRITTEN their new sums before moving to the next offset.
        __syncthreads(); 
    }

    // 4. Write the final computed scan back to global memory
    if (gid < n) {
        odata[gid] = temp[tid];
    }
}

