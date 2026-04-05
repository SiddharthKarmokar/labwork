#include<stdio.h>
#include<omp.h>
#include<time.h>

int main(){
    time_t now;
    time(&now);
    int nthreads = 4, tid;
    double start = omp_get_wtime();
    #pragma omp parallel num_threads(nthreads) private(tid)
    {
        tid = omp_get_thread_num();
        printf("Hello from %d\n", tid);
        #pragma omp for
        for(int i = 0; i < nthreads; i++){
            double ctime = omp_get_wtime();
            printf("For Hello from %d at time %f\n", tid, ctime - start);
        }
    }
    double end = omp_get_wtime();
    printf("It took %f seconds\n", end - start);
    return 0;
}