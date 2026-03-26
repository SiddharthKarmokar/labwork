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

void await_request(int rank, MPI_Request request){
    int waitcount = 0;
    int flag = 0;
    MPI_Status status;
    // do{
    //     waitcount++;
    //     MPI_Test(&request, &flag, &status);
    // }while(!flag);
    // MPI_Wait(&request, &status);
    debug(rank, "tested %d times\n", waitcount);
}

void send_to_many(int size){
    long randval = rand() / (RAND_MAX / 100);
    long value; 
    MPI_Request request;
    debug(0, "Sending %d to all processes\n", randval);
    for(int r = 1; r < size; r++){
        value = randval + r;
        MPI_Isend((void *)&value, 1, MPI_LONG, r, 1, MPI_COMM_WORLD, &request);
        // await_request(0, request);
        debug(0, "send %ld to %d\n", value, r);
    }
}

void receive_from_one(int rank, int size){
    long buffer;
    MPI_Request request;
    MPI_Status status;
    MPI_Irecv((void *)&buffer, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
    await_request(rank, request);
    debug(rank, "got value %ld\n", buffer);
}

int main(int argc, char**argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    debug(rank, "Hello (p=%d)\n", rank);
    if (rank == 0) {
        send_to_many(size);
    }else{
        receive_from_one(rank, size);
    }
    debug(rank, "Goodbye (p=%d)\n", rank);
    MPI_Finalize();
    return 0;
}