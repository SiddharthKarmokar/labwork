#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------
// Macros & Constants
// ---------------------------------------------------------
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// Padding macro to resolve shared memory bank conflicts
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// Macro for error checking CUDA API calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------------------------------------------------
// 1. The Highly Optimized CUDA Kernel
// ---------------------------------------------------------
__global__ void blelloch_scan_optimized(float *d_out, float *d_in, int n) {
    // Dynamic shared memory allocation
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    // Load data into shared memory with padding
    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankOffsetA] = d_in[ai];
    temp[bi + bankOffsetB] = d_in[bi];

    // Phase 1: Up-sweep (Reduce)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element to zero (Exclusive Scan requires this)
    if (thid == 0) {
        int last_idx = n - 1;
        temp[last_idx + CONFLICT_FREE_OFFSET(last_idx)] = 0;
    }

    // Phase 2: Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results back to global memory
    d_out[ai] = temp[ai + bankOffsetA];
    d_out[bi] = temp[bi + bankOffsetB];
}

// ---------------------------------------------------------
// 2. CPU Baseline for Verification
// ---------------------------------------------------------
void cpu_exclusive_scan(float *h_out, float *h_in, int n) {
    h_out[0] = 0; // Exclusive scan starts with 0
    for (int i = 1; i < n; i++) {
        h_out[i] = h_out[i - 1] + h_in[i - 1];
    }
}

// ---------------------------------------------------------
// 3. Main Driver
// ---------------------------------------------------------
int main() {
    // Note: This single-block implementation is limited to 2 * MAX_THREADS_PER_BLOCK
    // Modern GPUs support 1024 threads per block, so max N = 2048.
    const int N = 2048; 
    size_t size = N * sizeof(float);

    printf("--- CUDA Blelloch Prefix Scan (Optimized) ---\n");
    printf("Array Size: %d elements\n", N);

    // Allocate Host Memory
    float *h_in = (float *)malloc(size);
    float *h_out_gpu = (float *)malloc(size);
    float *h_out_cpu = (float *)malloc(size);

    // Initialize Host Array with 1s (Makes it easy to verify: sum should be 0, 1, 2, 3...)
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }

    // Allocate Device Memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Copy data to Device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Setup Execution Configuration
    int threadsPerBlock = N / 2; // Blelloch handles 2 elements per thread
    int blocksPerGrid = 1;

    // Calculate Shared Memory Size dynamically (including the padding elements)
    int sharedMemElements = N + CONFLICT_FREE_OFFSET(N);
    size_t sharedMemBytes = sharedMemElements * sizeof(float);

    // Setup CUDA Events for precise kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch Kernel & Record Time
    cudaEventRecord(start);
    blelloch_scan_optimized<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(d_out, d_in, N);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate Elapsed Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to Host
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost));

    // Run CPU version for verification
    cpu_exclusive_scan(h_out_cpu, h_in, N);

    // Verify Results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_out_gpu[i], h_out_cpu[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification: SUCCESS!\n");
        printf("Kernel Execution Time: %f ms\n", milliseconds);
        printf("Sample Output (Last Element): GPU[%d] = %f\n", N-1, h_out_gpu[N-1]);
    } else {
        printf("Verification: FAILED!\n");
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return 0;
}