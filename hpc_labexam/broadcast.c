#include<stdio.h>
#include<mpi.h>
#include<time.h>
#include<stdlib.h>
#include<stdarg.h>
#include<unistd.h>

MPI_Comm rowcom;
int worldrank, worldsize;

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

long broadcast(int rank, int size){
    long randomval;
    int broadcaster_rank = size - 1;
    if (rank == broadcaster_rank) {
        srand(time(NULL) + worldrank);
        randomval = rand() / (RAND_MAX / 10);
        debug(rank, "broadcasting %ld\n", randomval);
    }

    MPI_Bcast((void *)&randomval, 1, MPI_LONG, broadcaster_rank, rowcom);

    debug(rank, "received %ld\n", randomval);
    return randomval;
}

void barrier(int rank, long randomval){
    int naptime = randomval + (2 * worldrank);
    debug(rank, "sleeping %ds\n", naptime);
    sleep(naptime);

    debug(worldrank, "enter b-a-r-r-i-e-r\n");
    MPI_Barrier(rowcom);
    debug(worldrank, "leave barrier\n");
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldsize);
    int color = worldrank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldrank, &rowcom);

    int rank, size;
    MPI_Comm_rank(rowcom, &rank);
    MPI_Comm_size(rowcom, &size);
    printf("WORLD Rank %d \t| Color: %d \t| NEW Comm Rank: %d (out of %d)\n", 
           worldrank, color, rank, size);
    debug(rank, "hello (c=%d, p=%d)\n", color, rank);
    long randomval = broadcast(rank, size);
    barrier(rank, randomval);
    debug(rank, "goodbye\n");
    MPI_Finalize();
    return 0;
}