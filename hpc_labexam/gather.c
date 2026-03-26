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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int root = 0;
    int sendarray[3];
    for(int i = 0; i < 3; i++){
        sendarray[i] = (rank * 10) + i;
    }

    int *recvarray = NULL;
    if(rank == root) {
        int total_elements = size * 3;
        recvarray = (int*)malloc(total_elements * sizeof(int));
    }

    MPI_Gather(
        sendarray, 3, MPI_INT,
        recvarray, 3, MPI_INT,
        root, MPI_COMM_WORLD
    );

    if(rank == root){
        printf("Root (Rank %d) gathered the following data in order:\n", rank);
        for (int i = 0; i < size * 3; i++) {
            printf("%d ", recvarray[i]);
        }
        printf("\n");
        
        // Always free dynamically allocated memory
        free(recvarray);
    }

    MPI_Finalize();
    return 0;
}