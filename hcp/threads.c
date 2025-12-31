#include<stdio.h>
#include <stdlib.h>
#include<pthread.h>
#include<time.h>
#include<limits.h>

long time_diff_ns(struct timespec a, struct timespec b);

typedef  struct {
    int *arr;
    int start;
    int end;
    int max;
} ThreadData;


void* worker(void *args){
    ThreadData* data = (ThreadData*)args;
    int m = INT_MIN;
    for(int i = data->start; i < data->end; i++){
        if(data->arr[i] > m){
            m = data->arr[i];
        }
    }
    data->max = m;
    return NULL;
}

long time_diff_ns(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000000000L +
           (b.tv_nsec - a.tv_nsec);
}

int main(){
    int arr[1000];
    for(int i = 0; i < 1000; i++){
        arr[i] = rand();
    }
    
    pthread_t t1, t2;
    ThreadData d1, d2;
    d1.arr = arr;
    d1.start = 0;
    d1.end = 500;

    d2.arr = arr;
    d2.start = 500;
    d2.end = 1000;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_create(&t1, NULL, worker, &d1);
    pthread_create(&t2, NULL, worker, &d2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    int max = d1.max > d2.max ? d1.max : d2.max;
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Max element: %d\n", max);
    printf("Time taken: %ld ns\n", time_diff_ns(start, end));
    return 0;
}