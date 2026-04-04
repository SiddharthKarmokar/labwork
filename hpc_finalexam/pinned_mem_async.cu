#include <stdio.h>

// A dummy kernel
__global__ void addKernel(float *d_data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_data[tid] += 10.0f;
    }
}

int main() {
    int elements = 1000000;
    int size = elements * sizeof(float);
    
    float *h_data; // CPU pointer
    float *d_data; // GPU pointer

    // 1. Allocate Pinned Memory on the CPU (Host)
    cudaMallocHost((void**)&h_data, size);

    // 2. Allocate Device Memory on the GPU
    cudaMalloc((void**)&d_data, size);

    // Initialize some data on the CPU
    for (int i = 0; i < elements; i++) {
        h_data[i] = 1.0f;
    }

    // 3. Create a CUDA Stream
    // Think of this as a dedicated conveyor belt to the GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 4. Asynchronous Copy (Host to Device)
    // We pass the 'stream' as the final argument.
    // The CPU issues this command and instantly moves to the next line.
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

    // 5. Asynchronous Kernel Launch
    // We pass the stream in the 4th parameter of the execution configuration:
    // <<< grid, block, shared_memory, stream >>>
    int blockSize = 256;
    int gridSize = (elements + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize, 0, stream>>>(d_data, elements);

    // --- AT THIS EXACT MOMENT ---
    // The CPU is free! It can do independent math, read files, etc., 
    // while the GPU is handling the copy and the kernel in the background.
    printf("CPU is doing independent work here while GPU is busy!\n");

    // 6. Asynchronous Copy back to Host
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

    // 7. Synchronize
    // Now the CPU must stop and wait for everything in the stream to finish 
    // before it tries to print or use the results in 'h_data'.
    cudaStreamSynchronize(stream);

    // Clean up
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(stream);

    return 0;
}