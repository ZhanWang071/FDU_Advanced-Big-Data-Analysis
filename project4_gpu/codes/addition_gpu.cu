//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// kernel routine
// 

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}


//
// main code
//

int main(int argc, char **argv)
{
    cudaSetDevice(1);
    // Input the vector length
    int N = atoi(argv[1]);

    // Number of bytes to allocate for N float
    size_t bytes = N*sizeof(float);

    // Generate randomly vectors A and B
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);
    
    // Allocate memory for arrays d_A, d_B, and d_C on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    for (int i = 0; i < N; i++)
    {
        A[i] = rand()%100;
        B[i] = rand()%100;
    }

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Kernel invocation
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    int s = 0;
    for (int j = 0; j < N; j++) s += C[j];
        
    printf("\nGPU Vector Length: %d Sum: %d\n", N, s);

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();

    return 1;
}