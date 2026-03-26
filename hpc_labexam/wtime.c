#include "mpi.h"
#include <time.h>
#include <stdio.h>
/*measure_time.c*/
int main( int argc, char *argv[] )
{
double t1, t2;

MPI_Init( argc, argv);
t1 = MPI_Wtime();
sleep(1);
t2 = MPI_Wtime();
printf("MPI_Wtime measured a 1 second sleep to be: %1.2f\n", t2-t1);
fflush(stdout);
MPI_Finalize( );
return 0;
23 }