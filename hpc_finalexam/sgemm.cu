#include<stdio.h>
#include<cuda_runtime.h>

#define TILE_SIZE 16

__global__ void sgemmTiled(float *a, float *b, float *c, int N){
    
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * TILE_SIZE;
    int col = tx + blockIdx.x * TILE_SIZE;

    int numtiles = (N + TILE_SIZE - 1)/TILE_SIZE;
    float sum = 0.0f;
    for(int t = 0; t < numtiles; t++){
        int i = t * TILE_SIZE + tx; 
        int j = t * TILE_SIZE + ty; 
        if(row < N && i < N){
            tileA[ty][tx] = a[row * N + i];
        }else{
            tileA[ty][tx] = 0.0f;
        }

        if(col < N && j < N){
            tileB[ty][tx] = b[j * N + col];
        }else{
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();
        for(int i = 0; i < TILE_SIZE; i++){
            sum += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }
    if(row < N && col < N){
        c[row * N + col] = sum;
    }
}

int main(){
    float *a, *b, *c;
    int N = 8;
    size_t size = N * N * sizeof(float);
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            a[i*N + j] = j + 1;
            b[i*N + j] = j + 1;
        }
    }
    dim3 blocksPerGrid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    sgemmTiled<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);
    cudaDeviceSynchronize();
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", c[i*N + j]);
        }
        printf("\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}