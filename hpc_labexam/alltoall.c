#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ELEMENTS_PER_PROC 1

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Setup the SEND buffer
    // Total size = (Number of processes) * (Elements per process)
    int total_elements = size * ELEMENTS_PER_PROC;
    int *send_array = (int *)malloc(total_elements * sizeof(int));
    
    // Fill the send array. 
    // Format: (My Rank * 10) + (Destination Rank)
    // E.g., Rank 2's array for 4 processes: [20, 21, 22, 23]
    for (int dest = 0; dest < size; dest++) {
        send_array[dest] = (rank * 10) + dest; 
    }

    // 2. Setup the RECEIVE buffer
    int *recv_array = (int *)malloc(total_elements * sizeof(int));

    // 3. The All-To-All Operation!
    // NO root parameter, everyone participates equally.
    MPI_Alltoall(
        send_array, ELEMENTS_PER_PROC, MPI_INT, // Send buffer, count per dest, type
        recv_array, ELEMENTS_PER_PROC, MPI_INT, // Recv buffer, count per src, type
        MPI_COMM_WORLD
    );

    // 4. Print the results clearly
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Rank %d sent: [ ", rank);
    for (int i = 0; i < total_elements; i++) {
        printf("%02d ", send_array[i]);
    }
    printf("]  --->  Received: [ ");
    for (int i = 0; i < total_elements; i++) {
        printf("%02d ", recv_array[i]);
    }
    printf("]\n");

    // 5. Clean up
    free(send_array);
    free(recv_array);
    MPI_Finalize();
    return 0;
}