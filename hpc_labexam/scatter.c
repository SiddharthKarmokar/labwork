#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ELEMENTS_PER_PROC 2

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int root = 0;

    // 1. Setup the SEND buffer (ONLY for the Root process)
    int *send_array = NULL;
    if (rank == root) {
        int total_elements = size * ELEMENTS_PER_PROC;
        send_array = (int *)malloc(total_elements * sizeof(int));
        
        // Fill the array with values: 0, 1, 2, 3, 4, 5...
        printf("Root is creating the deck: [ ");
        for (int i = 0; i < total_elements; i++) {
            send_array[i] = i;
            printf("%d ", send_array[i]);
        }
        printf("]\n\n");
    }

    // 2. Setup the RECEIVE buffer for ALL processes
    // Every process needs a small array to hold their specific chunk
    int *recv_array = (int *)malloc(ELEMENTS_PER_PROC * sizeof(int));

    // 3. The Scatter Operation!
    // EVERY process calls this line.
    MPI_Scatter(
        send_array, ELEMENTS_PER_PROC, MPI_INT, // What Root is sending (ignored by others)
        recv_array, ELEMENTS_PER_PROC, MPI_INT, // Where everyone is saving their chunk
        root, MPI_COMM_WORLD                    // Who is dealing the data
    );

    // 4. Print the result on ALL processes
    // (Using a sleep to keep the terminal output relatively ordered)
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Process %d received: [ ", rank);
    for (int i = 0; i < ELEMENTS_PER_PROC; i++) {
        printf("%d ", recv_array[i]);
    }
    printf("]\n");

    // 5. Clean up memory
    if (rank == root) {
        free(send_array);
    }
    free(recv_array);

    MPI_Finalize();
    return 0;
}