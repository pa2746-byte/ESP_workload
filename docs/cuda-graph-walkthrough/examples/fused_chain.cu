// Fused version: the intermediate sum is an expression, not a global buffer.
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fusedAdd(const float *A, const float *B, float *D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        D[i] = (A[i] + B[i]) + 5.0f;
    }
}

int main() {
    const int N = 8;
    const size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 3.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B;
    float *d_D;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_D, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 4;
    int blocks = (N + threads - 1) / threads;

    fusedAdd<<<blocks, threads>>>(d_A, d_B, d_D, N);

    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("D[0] = %f\n", h_D[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    free(h_A);
    free(h_B);
    free(h_D);

    return 0;
}
