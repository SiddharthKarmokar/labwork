#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>

void mpi_spmv_csr(int local_rows, int total_cols,
                  const double *val, const int *col_ind, const int *row_ptr,
                  const double *local_x, const int *recvcounts, const int *displs,
                  double *local_y, MPI_Comm comm){
    double *global_x = (double *)malloc(total_cols * sizeof(double));
    if(global_x == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(comm, 1);
    }
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Allgatherv(local_x, recvcounts[rank], MPI_DOUBLE,
                    global_x, recvcounts, displs, MPI_DOUBLE, comm);
    for(int i = 0; i < local_rows; i++){
        double sum = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            sum += val[j] * global_x[col_ind[j]];
        }
        local_y[i] = sum;
    }

    free(global_x);
}


int main(int argc, char**argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int total_cols = 4, local_rows = 2;
    double *val = NULL;
    int *col_ind = NULL;
    int *row_ptr = NULL;
    double *local_x = NULL;
    double *local_y = (double*)malloc(local_rows * sizeof(double));
    if(rank == 0){
        double v[] = {10.0, 20.0, 5.0};
        int c[] = {0, 1, 2};
        int r[] = {0, 1, 3};
        double lx[] = {1,0, 2.0};

        val = (double *)malloc(3 * sizeof(double));
        col_ind = (int *)malloc(3 * sizeof(int));
        row_ptr = (int *)malloc(3 * sizeof(int));
        local_x = (double *)malloc(2 * sizeof(double));

        for(int i=0; i<3; i++) { val[i] = v[i]; col_ind[i] = c[i]; row_ptr[i] = r[i]; }
        for(int i=0; i<2; i++) { local_x[i] = lx[i]; }
    }else if(rank == 1){
        double v[] = {30.0, 40.0};
        int c[] = {2, 3};
        int r[] = {0, 1, 2};
        double lx[] = {3.0, 4.0};

        val = (double *)malloc(3 * sizeof(double));
        col_ind = (int *)malloc(3 * sizeof(int));
        row_ptr = (int *)malloc(3 * sizeof(int));
        local_x = (double *)malloc(2 * sizeof(double));
    
        for(int i=0; i<3; i++) { val[i] = v[i]; col_ind[i] = c[i]; row_ptr[i] = r[i]; }
        for(int i=0; i<2; i++) { local_x[i] = lx[i]; }
    }
    int recvcounts[2] = {2, 2};
    int displs[2] = {0, 2};

    mpi_spmv_csr(local_rows, total_cols, val, col_ind, row_ptr,
                local_x, recvcounts, displs, local_y, MPI_COMM_WORLD);
    
    for (int p = 0; p < size; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            printf("Rank %d computed local_y:\n", rank);
            for (int i = 0; i < local_rows; i++) {
                printf("  y[%d] = %.1f\n", (rank * local_rows) + i, local_y[i]);
            }
        }
    }

    free(val);
    free(col_ind);
    free(row_ptr);
    free(local_x);
    free(local_y);

    MPI_Finalize();
    return 0;
}