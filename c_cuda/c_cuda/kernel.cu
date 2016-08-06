
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cstring>

#define SET_BIT_CHAR_ARRAY(arr, i) ( arr[i/8] |= 1 << (i % 8) )
#define CLEAR_BIT_CHAR_ARRAY(arr, i) ( arr[i/8] &= ~(char)(1 << (i % 8)) )
#define READ_BIT_CHAR_ARRAY(arr, i) ( arr[i/8] & 1 << (i % 8) )


//WARNING, has to be (x + 1) % cMaxThreads == 0
const int cMaxThreads = 1024;
const int cMax = 2047;
const int cNumBlocks = (cMax + 1) / cMaxThreads;
const int cResults = cMax / 2 + 1;
const int cResultCharArraySize = cResults / 8;

/*
#if (((cMax + 1) % cMaxThreads ) != 0)
	#pragma error
#endif */

// memory layout: index * 2 + 1
__global__ void calculatePrimeSingle(char *isPrime) 
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int i = blockDim.x * y + x;
	int numToCheck = i * 2 + 1;
	SET_BIT_CHAR_ARRAY(isPrime, i);
	for (int j = 3; j < (numToCheck / 2) && READ_BIT_CHAR_ARRAY(isPrime, i); j += 2) {
		if ((numToCheck % j) == 0) {
			CLEAR_BIT_CHAR_ARRAY(isPrime, i);
		}
	}
}

__global__ void clearAll(char *isPrime)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int i = blockDim.x * y + x;
	CLEAR_BIT_CHAR_ARRAY(isPrime, i);
}

__global__ void setAll(char *isPrime)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int i = blockDim.x * y + x;
	SET_BIT_CHAR_ARRAY(isPrime, i);
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t calculatePrimeCuda(char *isPrime)
{
    char *dev_isPrime = 0;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_isPrime, cMax * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_isPrime, isPrime, cResultCharArraySize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
	}


    // Launch a kernel on the GPU with one thread for each element.
	//calculatePrimeSingle <<<cNumBlocks, cMaxThreads >> >(dev_isPrime);

	clearAll << <cNumBlocks, cMaxThreads >> >(dev_isPrime);
	//setAll << <cNumBlocks, cMaxThreads >> >(dev_isPrime);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(isPrime, dev_isPrime, cResultCharArraySize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_isPrime);
    
    return cudaStatus;
}


int main()
{
	char *cIsPrime = (char *)malloc(cResultCharArraySize);
	memset((void*)cIsPrime, 0x00, cResultCharArraySize);

	clock_t tStart = clock();
	cudaError_t cudaStatus = calculatePrimeCuda(cIsPrime);
	clock_t tEnd = clock();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	int numPrimeNumbers = 0;
	for (int i = 0; i < cResults; i++) {
		if (READ_BIT_CHAR_ARRAY(cIsPrime, i)) {
			printf("%i, ", i * 2 + 1);
			numPrimeNumbers++;
		}
	}

	printf("are prime numbers.\nTotal %i prime numbers.\nElapsed: %f seconds\n", numPrimeNumbers, (double)(tEnd - tStart) / CLOCKS_PER_SEC);
	free(cIsPrime);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	
	getchar();

	return 0;
}