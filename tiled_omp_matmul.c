#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// ---------------------------------------------------------
// Architectural Parameters
// ---------------------------------------------------------
// BLOCK_SIZE tunes the code to your L1/L2 Cache size. 
// 64x64 or 128x128 elements (approx 32KB to 128KB) are usually ideal.
#define BLOCK_SIZE 64 

// ---------------------------------------------------------
// 1. The Highly Optimized Tiled Kernel
// ---------------------------------------------------------
void matmul_tiled_omp(double *A, double *B, double *C, int n) {
    
    // First, initialize C to 0 in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    // Outer Loops: Iterate over the TILES
    // collapse(2) groups the 2D grid of tiles into a single 1D list of tasks 
    // and distributes them evenly across all CPU cores.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {
            
            // The k-block loop stays inside so each thread accumulates a full tile of C
            for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
                
                // Handle edge cases where N is not perfectly divisible by BLOCK_SIZE
                int i_end = (i_block + BLOCK_SIZE > n) ? n : i_block + BLOCK_SIZE;
                int j_end = (j_block + BLOCK_SIZE > n) ? n : j_block + BLOCK_SIZE;
                int k_end = (k_block + BLOCK_SIZE > n) ? n : k_block + BLOCK_SIZE;

                // Inner Loops: Compute the specific tile
                // CRITICAL OPTIMIZATION: Loop order is i -> k -> j
                for (int i = i_block; i < i_end; i++) {
                    for (int k = k_block; k < k_end; k++) {
                        
                        // We pull this value out so we don't look it up repeatedly
                        double a_ik = A[i * n + k];
                        
                        // Force the compiler to use AVX/SIMD vector instructions here
                        #pragma omp simd
                        for (int j = j_block; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------
// 2. Main Driver & Benchmark
// ---------------------------------------------------------
int main() {
    // 2048 x 2048 is large enough to show the cache benefits 
    // but small enough that we don't have to wait 10 minutes.
    int n = 2048; 
    size_t bytes = n * n * sizeof(double);

    printf("--- OpenMP Tiled Matrix Multiplication ---\n");
    printf("Matrix Size: %d x %d\n", n, n);
    printf("Block Size:  %d\n", BLOCK_SIZE);
    printf("Threads:     %d\n\n", omp_get_max_threads());

    // Allocate memory as flat 1D arrays (Crucial for sequential memory access)
    // Using aligned_alloc can help SIMD instructions, but malloc is fine for general use
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Initialize matrices with random data
    printf("Initializing data...\n");
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    printf("Computing matrix multiplication...\n");
    double start_time = omp_get_wtime();
    
    // Call the optimized kernel
    matmul_tiled_omp(A, B, C, n);
    
    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    // Calculate Performance (GFLOPS)
    double total_ops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = (total_ops / time_taken) / 1e9;

    printf("----------------------------------------\n");
    printf("Execution Time: %f seconds\n", time_taken);
    printf("Performance:    %.2f GFLOPS\n", gflops);
    printf("Sample Result:  C[0] = %f\n", C[0]); // Print one element to prevent compiler from optimizing out

    // Cleanup
    free(A); free(B); free(C);

    return 0;
}