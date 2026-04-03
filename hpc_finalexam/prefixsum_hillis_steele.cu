#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

#define BLOCK_SIZE 1024

__global__ void prefixSum(int *a, int *b, int n){
	__shared__ int temp[BLOCK_SIZE];
	
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;
	
	if(gid < n){
		temp[tid] = a[gid];
	}else{
		temp[tid] = 0;
	}
	__syncthreads();
	for(int offset = 1; offset <= blockDim.x; offset *= 2){
		int val = 0;
		if(tid < n && tid >= offset){
			val = temp[tid - offset];
		}
		__syncthreads();
		if(tid >= offset){
			temp[tid] += val;
		}
		__syncthreads();
	}

	if(gid < n){
		b[gid] = temp[tid];
	}
}


int main(){
	int *a, *b;
	int n = 1024;
	size_t size = n * sizeof(int);
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	for(int i = 0; i < n; i++){
		a[i] = i + 1;
	}

	int *d_a, *d_b;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 1024;
	int blocksPerGrid = 1;
	
	prefixSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n);
	
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n; i++){
		printf("%d ", b[i]);
	}
	printf("\n");
	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	return 0;
}
