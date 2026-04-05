#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        // The wrapper: tells OpenMP we are about to list independent tasks
        #pragma omp sections
        {
            // Task 1
            #pragma omp section
            {
                printf("Task A executed by thread %d\n", omp_get_thread_num());
                // e.g., Read data from disk
            }

            // Task 2
            #pragma omp section
            {
                printf("Task B executed by thread %d\n", omp_get_thread_num());
                // e.g., Initialize the matrix
            }

            // Task 3
            #pragma omp section
            {
                printf("Task C executed by thread %d\n", omp_get_thread_num());
                // e.g., Setup network connections
            }
        } // Implicit barrier here: all threads wait until all sections are done
    }
    return 0;
}