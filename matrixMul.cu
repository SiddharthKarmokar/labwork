#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------
// Architectural Parameters (Exam Requirements)
// ---------------------------------------------------------
// TILE_WIDTH: Defines our Block Size (16x16 = 256 threads). 
// This perfectly aligns with the hardware Warp Size (32), 
// as 256 is an exact multiple of 32, ensuring 100% warp utilization.
#define TILE_WIDTH 16 

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
// 1. The GPU Optimal Kernel (Tiled using Shared Memory)
// ---------------------------------------------------------
__global__ void matrixMulTiled(float *A, float *B, float *C, int Width) {
    // Allocate Shared Memory for tiles (Fast L1 Cache equivalent)
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify global row and column for this thread
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float pValue = 0.0;

    // Loop over the tiles required to compute the C element
    for (int m = 0; m < (Width - 1) / TILE_WIDTH + 1; ++m) {
        
        // Load data into shared memory with bounds checking
        if (Row < Width && m * TILE_WIDTH + tx < Width)
            tileA[ty][tx] = A[Row * Width + m * TILE_WIDTH + tx];
        else
            tileA[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < Width && Col < Width)
            tileB[ty][tx] = B[(m * TILE_WIDTH + ty) * Width + Col];
        else
            tileB[ty][tx] = 0.0;

        // Synchronize to ensure all threads have loaded their elements
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; ++k) {
            pValue += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize again before loading the next tile over current data
        __syncthreads();
    }

    // Write computed value back to global memory
    if (Row < Width && Col < Width) {
        C[Row * Width + Col] = pValue;
    }
}

// ---------------------------------------------------------
// 2. CPU Baseline for Verification
// ---------------------------------------------------------
void matrixMulCPU(float *A, float *B, float *C, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float sum = 0.0;
            for (int k = 0; k < Width; ++k) {
                sum += A[row * Width + k] * B[k * Width + col];
            }
            C[row * Width + col] = sum;
        }
    }
}

// ---------------------------------------------------------
// 3. Main Driver
// ---------------------------------------------------------
int main() {
    // Define Sample Size (N x N Matrix)
    // 1024 is a good size: large enough to saturate SMs, small enough for CPU to verify quickly
    int Width = 1024; 
    size_t size = Width * Width * sizeof(float);

    printf("--- Tiled Matrix Multiplication Bench ---\n");
    printf("Matrix Size: %d x %d\n", Width, Width);
    printf("Block Size: %d x %d (%d threads)\n", TILE_WIDTH, TILE_WIDTH, TILE_WIDTH * TILE_WIDTH);

    // Allocate Host Memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    // Initialize Host Matrices with random floats between 0.0 and 1.0
    for (int i = 0; i < Width * Width; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define Execution Configuration (Grid and Block dimensions)
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Setup CUDA Events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch Kernel & Record Time
    cudaEventRecord(start);
    matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, Width);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate Elapsed Time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to Host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------
    // Verification & Performance Metrics
    // ---------------------------------------------------------
    printf("\nRunning CPU baseline for verification (This takes a moment)...\n");
    matrixMulCPU(h_A, h_B, h_C_cpu, Width);

    bool success = true;
    for (int i = 0; i < Width * Width; i++) {
        // Floating point math isn't perfectly associative. 
        // CPU and GPU accumulate in different orders, so use a small epsilon for tolerance.
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-2) { 
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_C_gpu[i], h_C_cpu[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification: SUCCESS! (GPU matches CPU)\n\n");
        
        // Calculate GFLOPS
        // Matrix Multiplication requires 2 * N^3 operations (1 multiply, 1 add per inner loop)
        double total_ops = 2.0 * (double)Width * (double)Width * (double)Width;
        double seconds = milliseconds / 1000.0;
        double gflops = (total_ops / seconds) / 1e9;

        printf("--- Performance Metrics ---\n");
        printf("Execution Time:   %f ms\n", milliseconds);
        printf("Throughput:       %.2f GFLOPS\n", gflops);
    } else {
        printf("Verification: FAILED!\n");
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return 0;
}