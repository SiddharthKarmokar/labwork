#include <stdio.h>
#include <mpi.h>

// 1. Define the struct required for MINLOC and MAXLOC
struct int_loc {
    int val;
    int rank;
};

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int root = 0;

    // ==========================================
    // Category A: Arithmetic Operations
    // ==========================================
    int my_val = rank + 1; // Ranks 0, 1, 2, 3 have values 1, 2, 3, 4
    int sum_res, prod_res, max_res, min_res;

    MPI_Reduce(&my_val, &sum_res,  1, MPI_INT, MPI_SUM,  root, MPI_COMM_WORLD);
    MPI_Reduce(&my_val, &prod_res, 1, MPI_INT, MPI_PROD, root, MPI_COMM_WORLD);
    MPI_Reduce(&my_val, &max_res,  1, MPI_INT, MPI_MAX,  root, MPI_COMM_WORLD);
    MPI_Reduce(&my_val, &min_res,  1, MPI_INT, MPI_MIN,  root, MPI_COMM_WORLD);

    // ==========================================
    // Category B: Logical and Bitwise Operations
    // ==========================================
    // For logicals: Rank 0=0(False), Rank 1=1(True), Rank 2=0, Rank 3=1
    int logic_val = (rank % 2); 
    int land_res, lor_res, lxor_res;
    int band_res, bor_res, bxor_res;

    // Logical (True/False evaluation)
    MPI_Reduce(&logic_val, &land_res, 1, MPI_INT, MPI_LAND, root, MPI_COMM_WORLD);
    MPI_Reduce(&logic_val, &lor_res,  1, MPI_INT, MPI_LOR,  root, MPI_COMM_WORLD);
    MPI_Reduce(&logic_val, &lxor_res, 1, MPI_INT, MPI_LXOR, root, MPI_COMM_WORLD);

    // Bitwise (Binary bit manipulation on my_val: 1, 2, 3, 4)
    MPI_Reduce(&my_val, &band_res, 1, MPI_INT, MPI_BAND, root, MPI_COMM_WORLD);
    MPI_Reduce(&my_val, &bor_res,  1, MPI_INT, MPI_BOR,  root, MPI_COMM_WORLD);
    MPI_Reduce(&my_val, &bxor_res, 1, MPI_INT, MPI_BXOR, root, MPI_COMM_WORLD);

    // ==========================================
    // Category C: Location Operations (MINLOC/MAXLOC)
    // ==========================================
    struct int_loc my_loc_data;
    // Math trick: (rank - 2)^2 creates values: 4, 1, 0, 1 ... so Rank 2 has the minimum.
    my_loc_data.val = (rank - 2) * (rank - 2); 
    my_loc_data.rank = rank;
    
    struct int_loc minloc_res, maxloc_res;
    
    // Notice the datatype is MPI_2INT because we are passing a struct with 2 integers!
    MPI_Reduce(&my_loc_data, &minloc_res, 1, MPI_2INT, MPI_MINLOC, root, MPI_COMM_WORLD);
    MPI_Reduce(&my_loc_data, &maxloc_res, 1, MPI_2INT, MPI_MAXLOC, root, MPI_COMM_WORLD);

    // ==========================================
    // Print Results (Only Root has the final answers)
    // ==========================================
    if (rank == root) {
        printf("--- Arithmetic (Inputs: 1, 2, 3, 4...) ---\n");
        printf("SUM: %d | PROD: %d | MAX: %d | MIN: %d\n\n", sum_res, prod_res, max_res, min_res);
        
        printf("--- Logical (Inputs: 0, 1, 0, 1...) ---\n");
        printf("LAND: %d | LOR: %d | LXOR: %d\n\n", land_res, lor_res, lxor_res);
        
        printf("--- Bitwise (Inputs: 1, 2, 3, 4...) ---\n");
        printf("BAND: %d | BOR: %d | BXOR: %d\n\n", band_res, bor_res, bxor_res);
        
        printf("--- Location (Values: 4, 1, 0, 1...) ---\n");
        printf("MINLOC: Value %d found at Rank %d\n", minloc_res.val, minloc_res.rank);
        printf("MAXLOC: Value %d found at Rank %d\n", maxloc_res.val, maxloc_res.rank);
    }

    MPI_Finalize();
    return 0;
}