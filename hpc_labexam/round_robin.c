#include<stdio.h>
#include<mpi.h>
#include <stdlib.h>
#include<time.h>

void round_robin(int rank, int size){
    long int rand_mine, rand_prev;
    int rank_next = (rank + 1)%size;
    int rank_prev = rank == 0 ? size - 1 : rank - 1;
    MPI_Status status;

    srand(time(NULL) + rank);
    rand_mine = rand() / (RAND_MAX / 100);
    printf("%d : random is %ld\n", rank, rand_mine);

    if(rank % 2 == 0){
        printf("%d : Sending %ld to %d\n", rank, rand_mine, rank_next);
        MPI_Ssend((void *)&rand_mine, 1, MPI_LONG, rank_next, 1, MPI_COMM_WORLD);

        MPI_Recv((void*)&rand_prev, 1, MPI_LONG, rank_prev, 1, MPI_COMM_WORLD, &status);
        printf("%d : Received %ld from %d\n", rank, rand_prev, rank_prev);
    }else{
        MPI_Ssend((void *)&rand_mine, 1, MPI_LONG, rank_next, 1, MPI_COMM_WORLD);
        printf("%d : Sending %ld to %d\n", rank, rand_mine, rank_next);
        
        MPI_Recv((void*)&rand_prev, 1, MPI_LONG, rank_prev, 1, MPI_COMM_WORLD, &status);
        printf("%d : Received %ld from %d\n", rank, rand_prev, rank_prev);
    }

    printf("%d: I had %ld, %d had %ld\n", rank, rand_mine, rank_prev, rand_prev);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("%d : Hello\n", rank);
    round_robin(rank, size);
    printf("%d : Goodbye\n", rank);
    MPI_Finalize();
    return 0;
}