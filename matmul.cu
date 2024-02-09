// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Add This definition if you want the final result to be checked
//#define CHECK_C

// Step 1 and Step 2 Solution 1: T_W = 16, B_D = 32
// Step 2 Solution 2,3: T_W = 4, B_D = 32
// Step 3: TW = 2, B_D = 16
#define TILE_WIDTH 2
#define BLOCK_DIM 16
#define N  TILE_WIDTH * BLOCK_DIM

__global__ void
matrixMulCUDA_1(float* C, float* A, float* B, int n)
{
	int row = threadIdx.x;
	int col = threadIdx.y;

	if (row >= n || col >= n)
		return;

	float sum = 0.0f;
	for (int k = 0; k < n; k++) {
		sum += A[row * n + k] * B[k * n + col];
	}
	C[row * n + col] = sum;
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
__global__ void
matrixMulCUDA_2_1(float *C, float *A, float *B, int n)
{
	int tile_width = (n - 1) / BLOCK_DIM + 1;
	int start_row = threadIdx.y * tile_width;
	int end_row = start_row + tile_width;

	int start_col = threadIdx.x * tile_width;
	int end_col = start_col + tile_width;

	if (end_row > n)
		end_row = n;
	if (end_col > n)
		end_col = n;

	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			float partial_sum = 0.0f;
			for (int k = 0; k < n; k++) {
				partial_sum += A[row * n + k] * B[k * n + col];
			}
			C[row * n + col] = partial_sum;
		}
	}
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
__global__ void
matrixMulCUDA_2_2(float* C, float* A, float* B, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row >= n || col >= n)
		return;

	float partial_sum = 0.0f;
	for (int k = 0; k < n; k++) {
		partial_sum += A[row * n + k] * B[k * n + col];
	}
	C[row * n + col] = partial_sum;
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
__global__ void
matrixMulCUDA_2_3(float* C, float* A, float* B, int n)
{
	int start_row = blockIdx.y * blockDim.y * TILE_WIDTH + threadIdx.y * TILE_WIDTH;
	int end_row = start_row + TILE_WIDTH;

	int start_col = blockIdx.x * blockDim.x * TILE_WIDTH + threadIdx.x * TILE_WIDTH;
	int end_col = start_col + TILE_WIDTH;

	if (end_row > n)
		end_row = n;
	if (end_col > n)
		end_col = n;

	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			float partial_sum = 0;
			for (int k = 0; k < n; k++) {
				partial_sum += A[row * n + k] * B[k * n + col];
			}
			C[row * n + col] = partial_sum;
		}
	}
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
__global__ void
matrixMulCUDA_3(float* C, float* A, float* B, int n)
{
	int start_row = blockIdx.y * N + threadIdx.y;
	int end_row = start_row + N;

	int start_col = blockIdx.x * N + threadIdx.x;
	int end_col = start_col + N;

	int start_k, end_k;

	if (end_row > n)
		end_row = n;
	if (end_col > n)
		end_col = n;

	__shared__ float ds_A[N * N];
	__shared__ float ds_B[N * N];

	// Zero Matrix C because it has to be used as an accumulator
	for (int row = start_row; row < end_row; row += BLOCK_DIM) {
		for (int col = start_col; col < end_col; col += BLOCK_DIM) {
			C[row * n + col] = 0;
		}
	}

	for (int t = 0; t < n; t += N) {
		// Copy BigBlock of A into Share Memory
		start_k = t + threadIdx.x;
		end_k = t + N;
		if (end_k > n)
			end_k = n;
		for (int row = start_row, int ds_row = threadIdx.y; row < end_row; row += BLOCK_DIM, ds_row += BLOCK_DIM) {
			for (int k = start_k, int ds_k = threadIdx.x; k < end_k; k += BLOCK_DIM, ds_k += BLOCK_DIM) {
				ds_A[ds_row * N + ds_k] = A[row * n + k];
			}
		}
		
		// Copy BigBlock of B into Share Memory
		start_k = t + threadIdx.y;
		end_k = t + N;
		if (end_k > n)
			end_k = n;
		for (int k = start_k, int ds_k = threadIdx.y; k < end_k; k += BLOCK_DIM, ds_k += BLOCK_DIM) {
			for (int col = start_col, int ds_col = threadIdx.x; col < end_col; col += BLOCK_DIM, ds_col += BLOCK_DIM) {
				ds_B[ds_k * N + ds_col] = B[k * n + col];
			}
		}

		__syncthreads();

		start_k = 0;
		end_k = t + BLOCK_DIM * TILE_WIDTH;
		if (end_k > n)
			end_k = n;
		end_k -= t;
		for (int row = start_row, int ds_row = threadIdx.y; row < end_row; row += BLOCK_DIM, ds_row += BLOCK_DIM) {
			for (int col = start_col, int ds_col = threadIdx.x; col < end_col; col += BLOCK_DIM, ds_col += BLOCK_DIM) {
				float partial_sum = 0;
				for (int k = 0; k < end_k; k++) {
					partial_sum += A[ds_row * n + k] * B[k * n + ds_col];
				}
				C[row * n + col] += partial_sum;
			}
		}

		__syncthreads();
	}
}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int argc, char **argv, int n)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float)* size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float)* size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

#ifdef CHECK_C
	printf("Final Result Shall be Checked.\n");
	printf("Calculating resulting matrix using simple method on CPU:\n");
	// Calculate Correct C matrix for final testing
	float* h_C_test = (float*)malloc(mem_size_C);
	if (h_C_test == NULL)
	{
		fprintf(stderr, "Failed to allocate host test matrix C!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			h_C_test[i * n + j] = 0;
			for (int k = 0; k < n; k++)
				h_C_test[i * n + j] += h_A[i * n + k] * h_B[k * n + j];
		}
	}
	printf("Resulting matrix calculated\n");
#endif


	cudaError_t error;
	
	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(BLOCK_DIM, BLOCK_DIM);

	// Use for step 1 and step 2 solution 1
	//dim3 grid(1,1);
	
	// Use for step 2 solution 2
	//dim3 grid(
	//	(n - 1) / BLOCK_DIM + 1,
	//	(n - 1) / BLOCK_DIM + 1);

	// Use for step 2 solution 3 and step 3
	dim3 grid(
		(n - 1) / (BLOCK_DIM * TILE_WIDTH) + 1,
		(n - 1) / (BLOCK_DIM * TILE_WIDTH) + 1);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	matrixMulCUDA_3 << < grid, threads >> > (d_C, d_A, d_B, n);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronise device (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("Elapsed time in msec = %f\n", msecTotal);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

#ifdef CHECK_C
	// Check against h_C_test
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (h_C[i * n + j] - h_C_test[i * n + j] > 0.1 || h_C[i * n + j] - h_C_test[i * n + j] < -0.1) {
				printf("Wrong Multiplication Result! Values: %f, %f @ (%d, %d)\n", h_C[i * n + j], h_C_test[i * n + j], i, j);
				exit(EXIT_FAILURE);
			}
	printf("Calcuated Matrix is Correct.\n");
	free(h_C_test);
#endif

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;

}


/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Size of square matrices
	size_t n = 0;
	printf("[-] N = ");
	scanf("%u", &n);

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", n, n, n, n);

	int matrix_result = matrixMultiply(argc, argv, n);

	exit(matrix_result);
}
