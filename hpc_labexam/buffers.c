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

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        int dc = 1;
        int messagesize = dc * sizeof(int);
        int buffersize = messagesize + MPI_BSEND_OVERHEAD;

        void *mybuffer = malloc(buffersize);
        MPI_Buffer_attach(mybuffer, buffersize);

        int ds = 42;
        printf("Rank 0: Attached buffer (%d bytes). B-Sending data: %d\n", buffersize, ds);

        MPI_Bsend(&ds, dc, MPI_INT, 1, 0, MPI_COMM_WORLD);

        void *detachedbuffer;
        int detachedsize;

        MPI_Buffer_detach(&detachedbuffer, &detachedsize);

        free(detachedbuffer);
        printf("Rank 0: Buffer detached and memory freed safely.\n");
    } else if(rank == 1){
        int recv;
        MPI_Status status;

        MPI_Recv(&recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: Received data: %d\n", recv);
    }

    MPI_Finalize();
    return 0;
}