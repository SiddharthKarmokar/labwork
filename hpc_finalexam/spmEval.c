#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <string.h>

#define N 5000000       // 5 Million Rows (Vector x is ~40MB, exceeds most L3 caches)
#define NNZ_PER_ROW 10  // 50 Million total non-zeros
#define TILE_SIZE 2048
#define NUM_THREADS 4

typedef struct {
    int *row_ptr;
    int *col_idx;
    double *values;
    int nrows;
    int nnz;
} CSRMatrix;

// ---------------------------------------------------------
// 1. Random Sparse Matrix Generator
// ---------------------------------------------------------
void generate_random_sparse(CSRMatrix *A, int n, int nnz_per_row) {
    A->nrows = n;
    A->nnz = n * nnz_per_row;
    A->row_ptr = (int *)malloc((n + 1) * sizeof(int));
    A->col_idx = (int *)malloc(A->nnz * sizeof(int));
    A->values = (double *)malloc(A->nnz * sizeof(double));

    int curr = 0;
    for (int i = 0; i < n; i++) {
        A->row_ptr[i] = curr;
        for (int j = 0; j < nnz_per_row; j++) {
            A->col_idx[curr] = rand() % n; // Random column = Cache Misses!
            A->values[curr] = 1.0;
            curr++;
        }
    }
    A->row_ptr[n] = curr;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ---------------------------------------------------------
// 2. Standard OpenMP
// ---------------------------------------------------------
void spmv_omp_standard(CSRMatrix *A, double *x, double *y) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < A->nrows; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
            sum += A->values[j] * x[A->col_idx[j]];
        }
        y[i] = sum;
    }
}

// ---------------------------------------------------------
// 3. Tiled OpenMP (Row Strip Mining)
// ---------------------------------------------------------
void spmv_omp_tiled(CSRMatrix *A, double *x, double *y) {
    // Process the matrix in chunks (tiles) to keep working sets in L2 cache
    #pragma omp parallel for schedule(dynamic) proc_bind(spread)
    for (int tile = 0; tile < A->nrows; tile += TILE_SIZE) {
        int end_row = (tile + TILE_SIZE > A->nrows) ? A->nrows : tile + TILE_SIZE;
        
        for (int i = tile; i < end_row; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
                sum += A->values[j] * x[A->col_idx[j]];
            }
            y[i] = sum;
        }
    }
}

// ---------------------------------------------------------
// Pthread Data Structures
// ---------------------------------------------------------
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    CSRMatrix *A;
    double *x;
    double *y;
    int use_affinity;
} ThreadData;

// ---------------------------------------------------------
// 4. Pthreads Worker (Handles both Standard and Optimized)
// ---------------------------------------------------------
void* spmv_pthread_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // OPTIMIZATION: CPU Affinity Pinning
    if (data->use_affinity) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(data->thread_id, &cpuset); // Pin to specific physical core
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    for (int i = data->start_row; i < data->end_row; i++) {
        double sum = 0.0;
        for (int j = data->A->row_ptr[i]; j < data->A->row_ptr[i+1]; j++) {
            sum += data->A->values[j] * data->x[data->A->col_idx[j]];
        }
        data->y[i] = sum;
    }
    return NULL;
}

void run_pthreads(CSRMatrix *A, double *x, double *y, int use_affinity) {
    pthread_t threads[NUM_THREADS];
    ThreadData tdata[NUM_THREADS];
    int rows_per_thread = A->nrows / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        tdata[i] = (ThreadData){i, i * rows_per_thread, (i+1) * rows_per_thread, A, x, y, use_affinity};
        if (i == NUM_THREADS - 1) tdata[i].end_row = A->nrows;
        pthread_create(&threads[i], NULL, spmv_pthread_worker, &tdata[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

// ---------------------------------------------------------
// Main Benchmark Driver
// ---------------------------------------------------------
int main() {
    CSRMatrix A;
    double start, end;
    
    printf("--- SpMV Benchmark Suite ---\n");
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Non-zeros per row: %d (Total: %d)\n", NNZ_PER_ROW, N * NNZ_PER_ROW);
    printf("Threads: %d\n\n", NUM_THREADS);
    
    printf("Generating massive random sparse matrix (This may take a moment)...\n");
    generate_random_sparse(&A, N, NNZ_PER_ROW);

    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) x[i] = 1.0;

    // --- 1. Standard OpenMP ---
    memset(y, 0, N * sizeof(double));
    start = get_time();
    spmv_omp_standard(&A, x, y);
    end = get_time();
    printf("[1] Standard OpenMP:      %.4f seconds\n", end - start);

    // --- 2. Tiled OpenMP ---
    memset(y, 0, N * sizeof(double));
    start = get_time();
    spmv_omp_tiled(&A, x, y);
    end = get_time();
    printf("[2] Row-Tiled OpenMP:     %.4f seconds\n", end - start);

    // --- 3. Standard Pthreads ---
    memset(y, 0, N * sizeof(double));
    start = get_time();
    run_pthreads(&A, x, y, 0); // 0 = No affinity
    end = get_time();
    printf("[3] Standard Pthreads:    %.4f seconds\n", end - start);

    // --- 4. Optimized Pthreads ---
    memset(y, 0, N * sizeof(double));
    start = get_time();
    run_pthreads(&A, x, y, 1); // 1 = Use affinity
    end = get_time();
    printf("[4] Optimized Pthreads:   %.4f seconds\n", end - start);

    // Cleanup
    free(A.row_ptr); free(A.col_idx); free(A.values);
    free(x); free(y);

    return 0;
}