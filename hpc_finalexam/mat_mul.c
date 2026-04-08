#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Standard local dense matrix multiplication: C = C + A * B
void local_matrix_multiply(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

/**
 * Cannon's Algorithm for 2D Dense Matrix Multiplication
 */
void mpi_cannon_gemm(int local_n, double *local_A, double *local_B, double *local_C, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int grid_dim = (int)sqrt(size);
    
    // 1. Create a 2D Cartesian process grid with wrap-around (torus)
    int dims[2] = {grid_dim, grid_dim};
    int periods[2] = {1, 1}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(comm, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    int left, right, up, down;
    int block_elements = local_n * local_n;
    MPI_Status status;

    // 2. Initial Skewing
    // Shift local_A to the left by 'my_row' steps
    MPI_Cart_shift(cart_comm, 1, -my_row, &right, &left);
    MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE, 
                         left, 1, right, 1, cart_comm, &status);

    // Shift local_B upwards by 'my_col' steps
    MPI_Cart_shift(cart_comm, 0, -my_col, &down, &up);
    MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE, 
                         up, 2, down, 2, cart_comm, &status);

    // 3. Main Compute and Shift Loop
    for (int step = 0; step < grid_dim; step++) {
        // Multiply local blocks and accumulate into local_C
        local_matrix_multiply(local_n, local_A, local_B, local_C);

        // Shift local_A left by 1 step for the next iteration
        MPI_Cart_shift(cart_comm, 1, -1, &right, &left);
        MPI_Sendrecv_replace(local_A, block_elements, MPI_DOUBLE, 
                             left, 1, right, 1, cart_comm, &status);

        // Shift local_B up by 1 step for the next iteration
        MPI_Cart_shift(cart_comm, 0, -1, &down, &up);
        MPI_Sendrecv_replace(local_B, block_elements, MPI_DOUBLE, 
                             up, 2, down, 2, cart_comm, &status);
    }

    // 4. Cleanup
    MPI_Comm_free(&cart_comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Enforce 4 processes for this specific 2x2 grid example
    if (size != 4) {
        if (rank == 0) {
            printf("Please run this program with exactly 4 MPI processes (e.g., mpirun -np 4 ./cannon)\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Global matrix is 4x4. The process grid is 2x2.
    // Therefore, each local block is 2x2.
    int local_n = 2;
    int block_size = local_n * local_n;

    double *local_A = (double *)malloc(block_size * sizeof(double));
    double *local_B = (double *)malloc(block_size * sizeof(double));
    double *local_C = (double *)calloc(block_size, sizeof(double)); // Initialize C to 0

    // --- Hardcode the initial 2D Distribution ---
    // Rank 0: Top-Left (Rows 0-1, Cols 0-1)
    if (rank == 0) {
        double A_vals[] = {1, 2, 5, 6};
        for(int i=0; i<4; i++) { local_A[i] = A_vals[i]; local_B[i] = 1.0; }
    }
    // Rank 1: Top-Right (Rows 0-1, Cols 2-3)
    else if (rank == 1) {
        double A_vals[] = {3, 4, 7, 8};
        for(int i=0; i<4; i++) { local_A[i] = A_vals[i]; local_B[i] = 1.0; }
    }
    // Rank 2: Bottom-Left (Rows 2-3, Cols 0-1)
    else if (rank == 2) {
        double A_vals[] = {9, 10, 13, 14};
        for(int i=0; i<4; i++) { local_A[i] = A_vals[i]; local_B[i] = 1.0; }
    }
    // Rank 3: Bottom-Right (Rows 2-3, Cols 2-3)
    else if (rank == 3) {
        double A_vals[] = {11, 12, 15, 16};
        for(int i=0; i<4; i++) { local_A[i] = A_vals[i]; local_B[i] = 1.0; }
    }

    // --- Perform Cannon's Multiplication ---
    mpi_cannon_gemm(local_n, local_A, local_B, local_C, MPI_COMM_WORLD);

    // --- Print Results Sequentially ---
    for (int p = 0; p < size; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            printf("Rank %d local_C block:\n", rank);
            printf("  [%.1f, %.1f]\n", local_C[0], local_C[1]);
            printf("  [%.1f, %.1f]\n", local_C[2], local_C[3]);
        }
    }

    // --- Cleanup ---
    free(local_A);
    free(local_B);
    free(local_C);

    MPI_Finalize();
    return 0;
}