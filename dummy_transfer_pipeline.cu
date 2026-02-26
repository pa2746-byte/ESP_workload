#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void vecAdd(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void scaleKernel(const float* in, float* out, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * in[i];
}

__global__ void copyKernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

int main(int argc, char** argv) {
  // Args: N iters mode
  // mode=0 => transfer once (H2D once + D2H once)
  // mode=1 => transfer per-iter (H2D + D2H each iteration)
  int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
  int iters = (argc > 2) ? std::atoi(argv[2]) : 50;
  int mode = (argc > 3) ? std::atoi(argv[3]) : 1;

  if (N <= 0 || iters <= 0 || (mode != 0 && mode != 1)) {
    std::fprintf(stderr, "Usage: %s [N] [iters] [mode 0|1]\n", argv[0]);
    return EXIT_FAILURE;
  }

  size_t bytes = static_cast<size_t>(N) * sizeof(float);

  float *h_a = nullptr, *h_b = nullptr, *h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, bytes));
  CUDA_CHECK(cudaMallocHost(&h_b, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));

  for (int i = 0; i < N; i++) {
    h_a[i] = std::sin(0.001f * i);
    h_b[i] = std::cos(0.001f * i);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_tmp = nullptr, *d_scaled = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
  CUDA_CHECK(cudaMalloc(&d_scaled, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  if (mode == 0) {
    nvtxRangePushA("H2D_once");
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
    nvtxRangePop();
  }

  CUDA_CHECK(cudaEventRecord(start, stream));

  for (int it = 0; it < iters; it++) {
    if (mode == 1) {
      nvtxRangePushA("H2D_per_iter");
      CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
      nvtxRangePop();
    }

    nvtxRangePushA("vecAdd");
    vecAdd<<<grid, block, 0, stream>>>(d_a, d_b, d_tmp, N);
    nvtxRangePop();

    nvtxRangePushA("scale");
    scaleKernel<<<grid, block, 0, stream>>>(d_tmp, d_scaled, 1.01f, N);
    nvtxRangePop();

    nvtxRangePushA("memcpy_kernel");
    copyKernel<<<grid, block, 0, stream>>>(d_scaled, d_out, N);
    nvtxRangePop();

    if (mode == 1) {
      nvtxRangePushA("D2H_per_iter");
      CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
      nvtxRangePop();
    }
  }

  if (mode == 0) {
    nvtxRangePushA("D2H_once");
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    nvtxRangePop();
  }

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double checksum = 0.0;
  for (int i = 0; i < 1024 && i < N; i++) checksum += h_out[i];

  double expectedTransferBytes =
      (mode == 0) ? (3.0 * bytes) : (iters * 3.0 * bytes);

  std::printf("Dummy pipeline complete\n");
  std::printf("N=%d, iters=%d, mode=%d (%s)\n", N, iters, mode,
              mode == 0 ? "transfer_once" : "transfer_per_iter");
  std::printf("Elapsed: %.3f ms\n", ms);
  std::printf("Expected host-device transfer bytes: %.0f\n", expectedTransferBytes);
  std::printf("Checksum(0:1024): %.6f\n", checksum);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_scaled));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
