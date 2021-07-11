//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// CPU addition
// 

void VecAdd(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
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

    for (int i = 0; i < N; i++)
    {
        A[i] = rand()%100;
        B[i] = rand()%100;
    }

    VecAdd(A, B, C, N);

    int s = 0;
    for (int j = 0; j < N; j++) s += C[j];
        
    printf("\nCPU Vector Length: %d Sum: %d\n", N, s);

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // CUDA exit -- needed to flush printf write buffer
    cudaDeviceReset();

    return 1;
}