#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // For this specific example, we need exactly 3 processes
    if (size != 3) {
        if (rank == 0) printf("Please run with exactly 3 processes: mpirun -np 3 ./v_example\n");
        MPI_Finalize();
        return 0;
    }

    // ---------------------------------------------------------
    // 1. Setup local data for each process (Variable chunks)
    // ---------------------------------------------------------
    int send_count;
    int *send_buffer;

    if (rank == 0) {
        send_count = 2; // Rank 0 has 2 items
        send_buffer = (int*)malloc(send_count * sizeof(int));
        send_buffer[0] = 100; send_buffer[1] = 101;
    } else if (rank == 1) {
        send_count = 3; // Rank 1 has 3 items
        send_buffer = (int*)malloc(send_count * sizeof(int));
        send_buffer[0] = 200; send_buffer[1] = 201; send_buffer[2] = 202;
    } else if (rank == 2) {
        send_count = 1; // Rank 2 has 1 item
        send_buffer = (int*)malloc(send_count * sizeof(int));
        send_buffer[0] = 300;
    }

    // ---------------------------------------------------------
    // 2. Master Node sets up the 'Vector' tracking arrays
    // ---------------------------------------------------------
    int *recv_buffer = NULL;
    int *recv_counts = NULL;
    int *displacements = NULL;

    if (rank == 0) {
        // Total items expected: 2 + 3 + 1 = 6
        recv_buffer = (int*)malloc(6 * sizeof(int)); 
        recv_counts = (int*)malloc(3 * sizeof(int));
        displacements = (int*)malloc(3 * sizeof(int));

        // Array detailing exactly how many items to expect from each rank
        recv_counts[0] = 2; 
        recv_counts[1] = 3;
        recv_counts[2] = 1;

        // Array detailing exactly WHERE in the recv_buffer to place those items
        displacements[0] = 0; // Rank 0's data starts at index 0
        displacements[1] = 2; // Rank 1's data starts at index 2 (after Rank 0's 2 items)
        displacements[2] = 5; // Rank 2's data starts at index 5 (after Rank 1's 3 items)
    }

    // ---------------------------------------------------------
    // 3. Execute the Variable Gather
    // ---------------------------------------------------------
    MPI_Gatherv(send_buffer, send_count, MPI_INT,           // What I am sending
                recv_buffer, recv_counts, displacements, MPI_INT, // Where the Master is storing it
                0, MPI_COMM_WORLD);

    // ---------------------------------------------------------
    // 4. Master prints the final stitched-together array
    // ---------------------------------------------------------
    if (rank == 0) {
        printf("Master gathered data successfully: [ ");
        for (int i = 0; i < 6; i++) {
            printf("%d ", recv_buffer[i]);
        }
        printf("]\n");
        
        free(recv_buffer); 
        free(recv_counts); 
        free(displacements);
    }

    free(send_buffer);
    MPI_Finalize();
    return 0;
}