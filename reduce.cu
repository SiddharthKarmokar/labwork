#include<stdio.h>
#include "cuda_runtime.h"

__global__ void reduce(int *input, int *output, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        sdata[tid] = input[i];
    }else{
        sdata[tid] = 0;
    }
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if(tid % (2*s) == 0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sdata[0];
}


int main(){
    int n = 1024;
    size_t size = n * sizeof(int);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t output_size = blocksPerGrid * sizeof(int);

    int *h_i = (int*)malloc(size);
    int *h_o = (int*)malloc(output_size);
    for(int i = 0; i < n; i++){
        h_i[i] = i + 1;
    }
    int *d_i, *d_o;
    cudaMalloc((void**)&d_i, size);
    cudaMalloc((void**)&d_o, output_size);
    
    cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice);

    reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_i, d_o, n);

    cudaMemcpy(h_o, d_o, output_size, cudaMemcpyDeviceToHost);
    long finalsum = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        finalsum += h_o[i];
    }
    printf("final: %ld\n", finalsum);
    free(h_i);
    free(h_o);
    cudaFree(d_i);
    cudaFree(d_o);

    return 0;
}