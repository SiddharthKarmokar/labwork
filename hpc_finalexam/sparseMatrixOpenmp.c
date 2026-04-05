#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

typedef struct {
    int *rowptr;
    int *colidx;
    double *values;
    int nrows;
    int nnz;
} CSRMatrix;


void generate_sparse_matrix(CSRMatrix *A, int n) {
    A->nrows = n;
    // For a tri-diagonal matrix, nnz is roughly 3n
    A->nnz = 3 * n - 2;
    A->rowptr = (int *)malloc((n + 1) * sizeof(int));
    A->colidx = (int *)malloc(A->nnz * sizeof(int));
    A->values = (double *)malloc(A->nnz * sizeof(double));

    int curr_nnz = 0;
    for (int i = 0; i < n; i++) {
        A->rowptr[i] = curr_nnz;
        for (int j = i - 1; j <= i + 1; j++) {
            if (j >= 0 && j < n) {
                A->colidx[curr_nnz] = j;
                A->values[curr_nnz] = 1.0; // Simple value for testing
                curr_nnz++;
            }
        }
    }
    A->rowptr[n] = curr_nnz;
}

void spmv_omp(CSRMatrix *A, double *x, double *y){
    #pragma omp parallel for schedule(guided, 16) proc_bind(spread)
    for(int i = 0; i < A->nrows; i++){
        double sum = 0.0;
        int rowstart = A->rowptr[i];
        int rowend = A->rowptr[i+1];
        
        #pragma omp simd reduction(+:sum)
        for (int j = rowstart; j < rowend; j++){
            sum += A->values[j] * x[A->colidx[j]];
        }
        y[i] = sum;
    }
}

int main() {
    int N = 1000000; // 1 Million rows
    CSRMatrix A;
    
    printf("Generating %d x %d Sparse Matrix...\n", N, N);
    generate_sparse_matrix(&A, N);

    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));

    // Initialize vector x
    for (int i = 0; i < N; i++) x[i] = 1.0;

    // --- Performance Timing ---
    printf("Starting SpMV with OpenMP...\n");
    double start = omp_get_wtime();
    
    spmv_omp(&A, x, y);
    
    double end = omp_get_wtime();
    printf("Execution Time: %f seconds\n", end - start);

    // Basic Validation: In a tri-diagonal matrix of 1s, 
    // most entries in y should be 3.0 (except edges).
    printf("Sample Result: y[500] = %f\n", y[500]);

    // Cleanup
    free(A.rowptr);
    free(A.colidx);
    free(A.values);
    free(x);
    free(y);

    return 0;
}