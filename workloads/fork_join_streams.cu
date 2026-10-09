// Two independent GPU branches, followed by an explicit event-based join.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checkCuda(cudaError_t status, const char *operation, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at line %d (%s): %s\n",
                     line, operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation, __LINE__)

__global__ void addBranch(const float *a, const float *b, float *sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sum[i] = a[i] + b[i];
}

__global__ void subtractBranch(const float *a, const float *b, float *difference, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) difference[i] = a[i] - b[i];
}

__global__ void joinBranches(const float *sum, const float *difference, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sum[i] * difference[i];
}

int main() {
    // Deliberately not divisible by the block size, exercising the bounds guards.
    const int n = (1 << 20) + 17;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n), result(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(3 + i % 7);
        b[i] = static_cast<float>(1 + i % 3);
    }

    float *d_a, *d_b, *d_sum, *d_difference, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, bytes));
    CUDA_CHECK(cudaMalloc(&d_difference, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, bytes));

    cudaStream_t addStream, subtractStream, joinStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&addStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&subtractStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&joinStream, cudaStreamNonBlocking));
    cudaEvent_t addDone, subtractDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&addDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&subtractDone, cudaEventDisableTiming));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    // Nonblocking streams do not inherit ordering from the default stream.
    // Complete both uploads before either branch can read the shared inputs.
    CUDA_CHECK(cudaDeviceSynchronize());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    addBranch<<<blocks, threads, 0, addStream>>>(d_a, d_b, d_sum, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(addDone, addStream));

    subtractBranch<<<blocks, threads, 0, subtractStream>>>(d_a, d_b, d_difference, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(subtractDone, subtractStream));

    // Both branches have been submitted before either join wait is queued.
    // These waits order device work without blocking the CPU between branches.
    CUDA_CHECK(cudaStreamWaitEvent(joinStream, addDone, 0));
    CUDA_CHECK(cudaStreamWaitEvent(joinStream, subtractDone, 0));
    joinBranches<<<blocks, threads, 0, joinStream>>>(d_sum, d_difference, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(joinStream));
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, bytes, cudaMemcpyDeviceToHost));

    // Integer-valued inputs keep these operations exact in float32.
    size_t mismatches = 0;
    for (int i = 0; i < n; ++i) {
        const float expected = (a[i] + b[i]) * (a[i] - b[i]);
        if (result[i] != expected) {
            if (mismatches == 0)
                std::fprintf(stderr, "Mismatch at %d: got %f, expected %f\n",
                             i, result[i], expected);
            ++mismatches;
        }
    }
    std::printf("E[0] = %.6f (expected 8.000000)\n", result[0]);
    std::printf("Validation: %s (%zu mismatches across %d elements)\n",
                mismatches ? "FAIL" : "PASS", mismatches, n);

    CUDA_CHECK(cudaEventDestroy(addDone));
    CUDA_CHECK(cudaEventDestroy(subtractDone));
    CUDA_CHECK(cudaStreamDestroy(addStream));
    CUDA_CHECK(cudaStreamDestroy(subtractStream));
    CUDA_CHECK(cudaStreamDestroy(joinStream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_difference));
    CUDA_CHECK(cudaFree(d_result));
    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
