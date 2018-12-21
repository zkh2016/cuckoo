#include <stdio.h>
#include <cuda.h>

#define N_THREAD (128*128)
#define N (1 << 29)

__global__ void kernel1(int *result){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int sum = 0;
	for(int i = tid*32; i < N; i += N_THREAD*32){
		int tmpsum = 0;
		for(int j = i; j < i+32; j++)
			tmpsum += j;
		sum += tmpsum;
	}
	result[tid] = sum;
}

__global__ void kernel2(int *result){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int sum = 0;
	for(int i = tid * 64; i < N; i += N_THREAD*64){
		int tmpsum = 0;
		for(int j = i; j < i + 64; j++)
			tmpsum += j;
		sum += tmpsum;
	}
	result[tid] = sum;
}
__global__ void kernel3(int *result){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int sum = 0;
	int tmp[64];
	for(int i = tid * 64; i < N; i += N_THREAD*64){
		int tmpsum = 0;
		tmp[0] = i;
		for(int j = i+1; j < i + 64; j++){
			tmp[j-i] = tmp[j - 1 - i] + j;
			tmpsum += tmp[j-i];
		}
		/*for(int j = 0; j < 64; j++)
			tmpsum += tmp[j];
			*/
		sum += tmpsum;
	}	
	result[tid] = sum;
}

int main(){

	int *result = (int*)malloc(sizeof(int) * N_THREAD);
	int *dev_result;
	cudaMalloc((void**)&dev_result, sizeof(int)*N_THREAD);

	for(int i = 0; i < 100; i++){
	kernel1<<<128, 128>>>(dev_result);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(result, dev_result, sizeof(int) * N_THREAD, cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < 100; i ++){
	kernel2<<<128, 128>>>(dev_result);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(result, dev_result, sizeof(int) * N_THREAD, cudaMemcpyDeviceToHost);

	for(int i = 0; i < 100; i ++){
	kernel3<<<128, 128>>>(dev_result);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(result, dev_result, sizeof(int) * N_THREAD, cudaMemcpyDeviceToHost);
	cudaFree(dev_result);
	free(result);
	return 0;
}
