#include <stdio.h>
#include <unistd.h> // For sleep()
#include "mpi.h"

int main(int argc, char* argv[]) {
    int rank;
    double start_time, end_time, local_elapsed, max_elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 1. Synchronize all processes so they start the "stopwatch" together
    MPI_Barrier(MPI_COMM_WORLD); 
    start_time = MPI_Wtime(); // [cite: 830]

    // 2. Simulate variable work loads
    // Rank 0 sleeps for 1 second, Rank 1 for 2 seconds, etc.
    sleep(rank + 1); 

    end_time = MPI_Wtime(); // [cite: 832]
    local_elapsed = end_time - start_time;

    printf("Rank %d finished in %f seconds.\n", rank, local_elapsed);

    // 3. Find the overall execution time (the time of the slowest node)
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTotal Parallel Execution Time: %f seconds\n", max_elapsed);
    }

    MPI_Finalize();
    return 0;
}