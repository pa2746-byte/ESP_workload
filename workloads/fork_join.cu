// fork_join.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ void vectorSub(
    const float *A,
    const float *B,
    float *D,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        D[i] = A[i] - B[i];
}

__global__ void vectorMul(
    const float *C,
    const float *D,
    float *E,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        E[i] = C[i] * D[i];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_E = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 3.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B;
    float *d_C, *d_D, *d_E;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);
    cudaMalloc(&d_E, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    vectorSub<<<blocks, threads>>>(d_A, d_B, d_D, N);

    vectorMul<<<blocks, threads>>>(d_C, d_D, d_E, N);

    cudaMemcpy(h_E, d_E, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("E[0] = %f\n", h_E[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

    free(h_A);
    free(h_B);
    free(h_E);

    return 0;
}
