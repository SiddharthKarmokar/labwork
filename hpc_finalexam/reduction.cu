#include <stdio.h>
#include <cuda_runtime.h>

__global__ void
interleaved_reduction_kernel(float* g_out, float* g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Dynamic shared memory allocation
    extern __shared__ float s_data[];

    // 2. Load data into shared memory (with bounds checking)
    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f;
    __syncthreads();

    // 3. Perform reduction in shared memory
    // Stride starts at 1 and doubles; threads participate if their index 
    // is a multiple of (2 * stride)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        
        // We use 'index' to ensure we only target the "starts" of the active pairs
        int index = 2 * stride * threadIdx.x;

        if (index < blockDim.x) {
            s_data[index] += s_data[index + stride];
        }
        
        // Synchronize after every step of the reduction
        __syncthreads();
    }

    // 4. Write the result of this block to global memory
    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}


// ================= KERNEL =================
__global__ void reduction_kernel(float* d_out, float* d_in,
                                 unsigned int size) {
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    // Load into shared memory
    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;
    __syncthreads();

    // Reduction (your version)
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((threadIdx.x & (2 * stride - 1)) == 0 &&
            threadIdx.x + stride < blockDim.x) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write result per block
    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

// ================= HOST REDUCTION =================
void reduction(float *d_out, float *d_in, int n_threads, int size) {
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    while (size > 1) {
        int n_blocks = (size + n_threads - 1) / n_threads;

        reduction_kernel<<<n_blocks, n_threads,
                           n_threads * sizeof(float)>>>(d_out, d_out, size);

        cudaDeviceSynchronize();
        size = n_blocks;
    }
}

// ================= MAIN =================
int main() {
    int N = 1024;                 // number of elements
    int threads = 256;

    float *h_in;
    float *d_in, *d_out;

    size_t size = N * sizeof(float);

    // Allocate host memory
    h_in = (float*)malloc(size);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;   // easy to verify: sum = N
    }

    // Allocate device memory
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copy to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Run reduction
    reduction(d_out, d_in, threads, N);

    // Copy result back
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("GPU Reduction Result = %f\n", result);

    // Verify on CPU
    float cpu_sum = 0;
    for (int i = 0; i < N; i++) {
        cpu_sum += h_in[i];
    }

    printf("CPU Sum = %f\n", cpu_sum);

    // Cleanup
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}