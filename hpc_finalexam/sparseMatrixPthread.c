#define _GNU_SOURCE
#include<pthread.h>
#include<sched.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef struct {
    int *row_ptr;
    int *col_idx;
    double *values;
    int nrows;
    int nnz;
} CSRMatrix;

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    CSRMatrix *A;
    double *x;
    double *y;
} ThreadData;

void generate_sparse_matrix(CSRMatrix *A, int n) {
    A->nrows = n;
    A->nnz = 3 * n - 2;
    A->row_ptr = (int *)malloc((n + 1) * sizeof(int));
    A->col_idx = (int *)malloc(A->nnz * sizeof(int));
    A->values = (double *)malloc(A->nnz * sizeof(double));

    int curr_nnz = 0;
    for (int i = 0; i < n; i++) {
        A->row_ptr[i] = curr_nnz;
        for (int j = i - 1; j <= i + 1; j++) {
            if (j >= 0 && j < n) {
                A->col_idx[curr_nnz] = j;
                A->values[curr_nnz] = 1.0; 
                curr_nnz++;
            }
        }
    }
    A->row_ptr[n] = curr_nnz;
}

void* spmv_worker(void *arg){
    ThreadData* data = (ThreadData*)arg;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    for(int i = data->start_row; i < data->end_row; i++){
        double sum = 0.0;
        for(int j = data->A->row_ptr[i]; j < data->A->row_ptr[i+1]; j++){
            sum += data->A->values[j] * data->x[data->A->col_idx[j]];
        }
        data->y[i] = sum;
    }
    return NULL;
}

void launch_pthreads(CSRMatrix *A, double *x, double *y, int num_threads){
    pthread_t threads[num_threads];
    ThreadData tdata[num_threads];
    int rows_per_thread = A->nrows / num_threads;
    for(int i = 0; i < num_threads; i++){
        tdata[i] = (ThreadData){i, i * rows_per_thread, (i+1) * rows_per_thread, A, x, y};
        if (i == num_threads - 1)tdata[i].end_row = A->nrows;
        pthread_create(&threads[i], NULL, spmv_worker, &tdata[i]);
    }

    for(int i = 0; i < num_threads; i++){
        pthread_join(threads[i], NULL);
    }
}

int main() {
    int N = 1000000;      // 1 Million rows
    int num_threads = 4; // Adjust based on your CPU cores
    CSRMatrix A;
    
    printf("Initializing %d x %d Sparse Matrix (Pthreads)...\n", N, N);
    generate_sparse_matrix(&A, N);

    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) x[i] = 1.0;

    // Timing with clock_gettime
    struct timespec start, end;
    printf("Launching %d threads with CPU affinity...\n", num_threads);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    launch_pthreads(&A, x, y, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Execution Time: %f seconds\n", time_taken);
    printf("Sample Result: y[500] = %f\n", y[500]);

    // Cleanup
    free(A.row_ptr); free(A.col_idx); free(A.values);
    free(x); free(y);

    return 0;
}