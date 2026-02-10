#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <pthread.h>
#include <time.h>

// --- Structures for Pthreads ---
typedef struct {
    int *arr;
    int start;
    int end;
    int result;
} ThreadData;

// --- Helper: Generate Data ---
void generate_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i; 
    }
}

// ============================================================
// Approach 1: OpenMP Implicit Loop
// ============================================================
bool omp_check_loop(int *arr, int n, int num_threads) {
    int is_sorted = 1;
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for shared(is_sorted) schedule(static)
    for (int i = 0; i < n - 1; i++) {
        if (!is_sorted) continue; 
        if (arr[i] > arr[i+1]) {
            #pragma omp atomic write
            is_sorted = 0;
        }
    }
    return is_sorted;
}

// ============================================================
// Approach 2: OpenMP Explicit Decomposition
// ============================================================
bool omp_check_explicit(int *arr, int n, int req_threads) {
    int global_sorted = 1;
    omp_set_num_threads(req_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        int chunk_size = n / num_threads;
        int start = tid * chunk_size;
        int end = (tid == num_threads - 1) ? n - 1 : start + chunk_size;

        for (int i = start; i < end - 1; i++) {
            if (global_sorted && arr[i] > arr[i+1]) {
                #pragma omp atomic write
                global_sorted = 0;
            }
        }

        if (global_sorted && tid < num_threads - 1) {
            if (arr[end-1] > arr[end]) {
                 #pragma omp atomic write
                 global_sorted = 0;
            }
        }
    }
    return global_sorted;
}

// ============================================================
// Approach 3: OpenMP Task Parallelism
// ============================================================
int check_recursive_task(int *arr, int start, int end) {
    if (end - start < 1000) {
        for (int i = start; i < end - 1; i++) {
            if (arr[i] > arr[i+1]) return 0;
        }
        return 1;
    }

    int mid = (start + end) / 2;
    int left_res = 1, right_res = 1;

    #pragma omp task shared(left_res)
    left_res = check_recursive_task(arr, start, mid);

    #pragma omp task shared(right_res)
    right_res = check_recursive_task(arr, mid, end);

    #pragma omp taskwait
    int boundary_res = (arr[mid-1] <= arr[mid]);
    return left_res && right_res && boundary_res;
}

bool omp_check_tasks(int *arr, int n, int num_threads) {
    int result = 1;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        result = check_recursive_task(arr, 0, n);
    }
    return result;
}

// ============================================================
// Approach 4: OpenMP SIMD (Note: Threads don't apply here directly)
// ============================================================
bool omp_check_simd(int *arr, int n) {
    int is_sorted = 1;
    #pragma omp simd reduction(&&:is_sorted)
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i+1]) is_sorted = 0;
    }
    return is_sorted;
}

// ============================================================
// Approach 5: Pthreads
// ============================================================
void* pthread_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->result = 1;
    for (int i = data->start; i < data->end - 1; i++) {
        if (data->arr[i] > data->arr[i+1]) {
            data->result = 0;
            return NULL;
        }
    }
    return NULL;
}

bool pthreads_check(int *arr, int n, int num_threads) {
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData *tdata = malloc(num_threads * sizeof(ThreadData));
    
    int chunk_size = n / num_threads;
    int is_sorted = 1;

    for (int i = 0; i < num_threads; i++) {
        tdata[i].arr = arr;
        tdata[i].start = i * chunk_size;
        int limit = (i + 1) * chunk_size;
        if (i == num_threads - 1) limit = n;
        tdata[i].end = limit;
        
        pthread_create(&threads[i], NULL, pthread_worker, &tdata[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        if (tdata[i].result == 0) is_sorted = 0;
        
        // Manual boundary check in main thread
        if (i < num_threads - 1) {
            int boundary_idx = (i + 1) * chunk_size;
            if (arr[boundary_idx - 1] > arr[boundary_idx]) is_sorted = 0;
        }
    }
    
    free(threads);
    free(tdata);
    return is_sorted;
}

// ============================================================
// Main Driver
// ============================================================
int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <size> <mode> <threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int mode = atoi(argv[2]);
    int threads = atoi(argv[3]);
    
    int *arr = (int*)malloc(n * sizeof(int));
    generate_array(arr, n);

    struct timespec start_wall, end_wall;
    clock_gettime(CLOCK_MONOTONIC, &start_wall);

    bool result;
    switch(mode) {
        case 1: result = omp_check_loop(arr, n, threads); break;
        case 2: result = omp_check_explicit(arr, n, threads); break;
        case 3: result = omp_check_tasks(arr, n, threads); break;
        case 4: result = omp_check_simd(arr, n); break; // Threads ignored
        case 5: result = pthreads_check(arr, n, threads); break;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_wall);

    double time_wall = (end_wall.tv_sec - start_wall.tv_sec) + 
                       (end_wall.tv_nsec - start_wall.tv_nsec) / 1e9;

    char* names[] = {"", "OMP_Loop", "OMP_Explicit", "OMP_Tasks", "OMP_SIMD", "Pthreads"};
    
    // Output: Method, Size, Threads, WallTime
    printf("%s,%d,%d,%.9f\n", names[mode], n, threads, time_wall);

    free(arr);
    return 0;
}