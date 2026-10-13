// reduction.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256

__global__ void reduceStage1(const float *input, float *partialSums, int N)
{
    __shared__ float shared[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load one element per thread into shared memory
    if (idx < N)
        shared[tid] = input[idx];
    else
        shared[tid] = 0.0f;

    __syncthreads();

    // Tree reduction inside each block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    // One partial sum per block
    if (tid == 0) {
        partialSums[blockIdx.x] = shared[0];
    }
}

__global__ void reduceStage2(const float *partialSums, float *result, int numBlocks)
{
    __shared__ float shared[THREADS_PER_BLOCK];

    int tid = threadIdx.x;

    if (tid < numBlocks)
        shared[tid] = partialSums[tid];
    else
        shared[tid] = 0.0f;

    __syncthreads();

    // Final reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared[0];
    }
}

int main()
{
    // Keep N small enough that number of Stage 1 blocks
    // fits into one Stage 2 block.
    const int N = 1 << 16;  // 65536 elements

    const size_t bytes = N * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    float h_result = 0.0f;

    // Initialize everything to 1, so expected sum = N
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input;
    float *d_partialSums;
    float *d_result;

    int numBlocks =
        (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(
        &d_partialSums,
        numBlocks * sizeof(float)
    );
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice
    );

    // Stage 1:
    // Each block reduces part of the input into one partial sum.
    reduceStage1<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_input,
        d_partialSums,
        N
    );

    // Stage 2:
    // Reduce all partial sums into one final value.
    reduceStage2<<<1, THREADS_PER_BLOCK>>>(
        d_partialSums,
        d_result,
        numBlocks
    );

    cudaMemcpy(
        &h_result,
        d_result,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaDeviceSynchronize();

    printf("Result   = %f\n", h_result);
    printf("Expected = %d\n", N);

    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_result);

    free(h_input);

    return 0;
}
