#include<stdio.h>
#include<mpi.h>
#include<time.h>
#include<stdlib.h>
#include<stdarg.h>
#include<unistd.h>

double get_timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void debug(int rank, char *format, ...){
    va_list args;
    va_start(args, format);
    printf("%12.6f|%2d | ",get_timer(), rank);
    vprintf(format, args);

    va_end(args);
}

int main(int argc, char**argv){
    MPI_Init(&argc, &argv); //Finalize
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int sd = rank * 10;
    int *recv = (int*)malloc(size * sizeof(int)); //free & array pointer

    MPI_Allgather(
        &sd, 1, MPI_INT,
        recv, 1, MPI_INT,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d has gathered the full array: [ ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", recv[i]);
    }
    printf("]\n");

    free(recv);
    MPI_Finalize();
    return 0;
}