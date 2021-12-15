
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

//__device__ int get_lx(int a, int b, const int* x, const int n) {
//	int cell_x;
//	if (a == 5 && b = 1)
//		return x[]
//	else if (a == 7 || a == 11 || a == 14)
//		cell_x = 1;
//	else if (a == 8 || a == 12)
//		cell_x = 2;
//	else
//		cell_x = 3;
//
//	//if (a == 6 || a	if (a == 6 || a == 10 || a == 13 || a == 15)
//	//	cell_x = ;
//	//else if (a == 7 || a == 11 || a == 14)
//	//	cell_x = 1;
//	//else if (a == 8 || a == 12 )
//	//	cell_x = 2;
//	//else
//	//	cell_x = 3;
//
//
//
//	return x[cell_x];
//}
//
//__device__ int get_ly(int a, int b, const int* y) {
//
//	return 1;
//}
//
//__device__ int get_r(int x, int y) {
//	return 1;
//}
//
//__device__ int f(int x, int y) {
//	return 1;
//}
//
//__global__ void matrixchainmultipilicationkernel(int *ST, const int n, const int* k, const int* x, const int* y) {
//	int j = threadIdx.x;
//
//	for (int i = n + 1; i < (n*(n + 1) / 2) + n - 2; i++) {
//		int ij = i - j + 1;
//		if (ij <= k[ij]) {
//			//(substep 1)
//			int vlx = get_lx(ij, j);
//			int vly = get_ly(ij, j);
//			//(substep 2)
//			int vr = get_r(ij, j);
//			//(substep 3)
//			int vs = f(vl, vr);
//			//(substep 4)
//			if (j == 1)
//				st[i - j + 1] = vs;
//			else
//				st[i - j + 1] = st[ij] ↓ vs;
//			// where vl, vr, and vs are local variables in a thread
//
//		}
//	}
//}


int MatrixChainMultiplication() {
	const int dimensions [6] = { 30, 35, 15, 5, 10, 20 };
	const int n = 5;

	const int size = n * ((n + 1) / 2);
	
	int *location;
	location = (int *)malloc(size * sizeof(int));
	printf("locations \n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < ((n + 1) / 2); j++) {
			printf("{%d} \t", (n * (i - 1) - (i - 2) * (i - 1) / 2) + (j - i));
			//printf("{%d} \t", location[i*n + j]);
		}
		printf("\n");
	}

	

	int *host_st;
	host_st = (int *)malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		host_st[i] = 0;
	}
    int k[] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4 };

	int *x;
	x = (int *)malloc(size * sizeof(int));
	int *y;
	y = (int *)malloc(size * sizeof(int));


	for (int i = 0; i < n - 1; i++) {
		x[i] = dimensions[i];
		y[i] = dimensions[i - 1];
	}

	

	return 1;
}

int MatrixChainMultiplication_BottomUp(int p[], const int n) {
	printf("Perform Matrix multiplication {%d} \n", n);
	int *matrix_table;
	matrix_table = (int *)malloc(n*n * sizeof(int));

	//int matrix_table[n][n];

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			matrix_table[i * n + j] = 0;


	for (int L = 2; L < n; L++) {
		for (int i = 1; i < n - L + 1; i++) {
			int j = i + L - 1;
			matrix_table[i * n + j] = INT_MAX;
			for (int k = i; k <= j - 1; k++) {
				int q = matrix_table[i * n + k] + matrix_table[(k + 1)*n + j] + (p[i - 1] * p[k] * p[j]);
				if (q < matrix_table[i * n + j])
					matrix_table[i * n + j] = q;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("{%d} \t", matrix_table[i * n + j]);
		}
		printf("\n");
	}
	return matrix_table[n + (n - 1)];
}


// Driver Code
int main() {
	int arr[] = { 30, 35, 15, 5, 10, 20, 25 };
	int size = sizeof(arr) / sizeof(arr[0]);
	printf("Minimum number of multiplications is {%d} \n", MatrixChainMultiplication_BottomUp(arr, size));
	printf("Minimum number of multiplications is {%d} \n", MatrixChainMultiplication());
	return 0;
}

//int main() {
//	const int arraySize = 5;
//	const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	int c[arraySize] = { 0 };
//
//	// Add vectors in parallel.
//	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addWithCuda failed!");
//		return 1;
//	}
//
//	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//		c[0], c[1], c[2], c[3], c[4]);
//
//	// cudaDeviceReset must be called before exiting in order for profiling and
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//
//	return 0;
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

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
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
