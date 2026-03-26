#include<stdio.h>
#include<mpi.h>
#include<time.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdarg.h>

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

int main(int argc, char **argv){ 
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int rec[5];

    if(rank == 0){
        MPI_Status status;
        MPI_Recv(rec, 10, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("%d\n", status.MPI_SOURCE);
        printf("%d\n", status.MPI_TAG);
        printf("%d\n", status.MPI_ERROR);
        int recv_count = 0;
        MPI_Get_count(&status, MPI_INT, &recv_count);
        printf("Received %d data\n", recv_count);
        for(int i = 0; i < recv_count; i++){
            printf("recv[%d]=%d\n", i, rec[i]);
        }
    }else {
        MPI_Send((void *)&send, 10, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;
}