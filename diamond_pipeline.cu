// Diamond dependency pattern:
//
//         H2D(d_in)
//        /          \
//  kernelA(d_a)  kernelB(d_b)     <- parallel, both read d_in
//        \          /
//       kernelMerge(d_out)         <- depends on both d_a and d_b
//              |
//           D2H(h_out)
//
// Stream layout:
//   stream_main : H2D, kernelMerge, D2H
//   stream_a    : kernelA
//   stream_b    : kernelB
//
// CUDA events enforce the cross-stream data dependencies:
//   ev_h2d_done  -> stream_a and stream_b wait before their kernels
//   ev_a_done    -> stream_main waits before kernelMerge
//   ev_b_done    -> stream_main waits before kernelMerge

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

// Branch A: scale each element by 2
__global__ void kernelA(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 2.0f * in[i];
}

// Branch B: square each element
__global__ void kernelB(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * in[i];
}

// Merge: sum the two branches — depends on both kernelA and kernelB outputs
__global__ void kernelMerge(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  int N     = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
  int iters = (argc > 2) ? std::atoi(argv[2]) : 50;

  if (N <= 0 || iters <= 0) {
    std::fprintf(stderr, "Usage: %s [N] [iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  size_t bytes = static_cast<size_t>(N) * sizeof(float);

  float *h_in = nullptr, *h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in,  bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  for (int i = 0; i < N; i++) h_in[i] = std::sin(0.001f * i);

  float *d_in = nullptr, *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  bytes));
  CUDA_CHECK(cudaMalloc(&d_a,   bytes));
  CUDA_CHECK(cudaMalloc(&d_b,   bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  cudaStream_t stream_main, stream_a, stream_b;
  CUDA_CHECK(cudaStreamCreate(&stream_main));
  CUDA_CHECK(cudaStreamCreate(&stream_a));
  CUDA_CHECK(cudaStreamCreate(&stream_b));

  cudaEvent_t ev_h2d_done, ev_a_done, ev_b_done;
  CUDA_CHECK(cudaEventCreate(&ev_h2d_done));
  CUDA_CHECK(cudaEventCreate(&ev_a_done));
  CUDA_CHECK(cudaEventCreate(&ev_b_done));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t t_start, t_stop;
  CUDA_CHECK(cudaEventCreate(&t_start));
  CUDA_CHECK(cudaEventCreate(&t_stop));
  CUDA_CHECK(cudaEventRecord(t_start, stream_main));

  for (int it = 0; it < iters; it++) {

    // ── Source node: H2D ─────────────────────────────────────────────────
    nvtxRangePushA("H2D");
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes,
                               cudaMemcpyHostToDevice, stream_main));
    nvtxRangePop();
    CUDA_CHECK(cudaEventRecord(ev_h2d_done, stream_main));

    // ── Branch A: kernelA reads d_in, writes d_a ─────────────────────────
    CUDA_CHECK(cudaStreamWaitEvent(stream_a, ev_h2d_done, 0));
    nvtxRangePushA("kernelA");
    kernelA<<<grid, block, 0, stream_a>>>(d_in, d_a, N);
    nvtxRangePop();
    CUDA_CHECK(cudaEventRecord(ev_a_done, stream_a));

    // ── Branch B: kernelB reads d_in, writes d_b ─────────────────────────
    CUDA_CHECK(cudaStreamWaitEvent(stream_b, ev_h2d_done, 0));
    nvtxRangePushA("kernelB");
    kernelB<<<grid, block, 0, stream_b>>>(d_in, d_b, N);
    nvtxRangePop();
    CUDA_CHECK(cudaEventRecord(ev_b_done, stream_b));

    // ── Sink node: wait for both branches, merge, then D2H ───────────────
    CUDA_CHECK(cudaStreamWaitEvent(stream_main, ev_a_done, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream_main, ev_b_done, 0));
    nvtxRangePushA("kernelMerge");
    kernelMerge<<<grid, block, 0, stream_main>>>(d_a, d_b, d_out, N);
    nvtxRangePop();

    nvtxRangePushA("D2H");
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, stream_main));
    nvtxRangePop();
  }

  CUDA_CHECK(cudaEventRecord(t_stop, stream_main));
  CUDA_CHECK(cudaEventSynchronize(t_stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));

  double checksum = 0.0;
  for (int i = 0; i < 1024 && i < N; i++) checksum += h_out[i];

  std::printf("Diamond pipeline complete\n");
  std::printf("N=%d, iters=%d\n", N, iters);
  std::printf("Elapsed: %.3f ms\n", ms);
  std::printf("Checksum(0:1024): %.6f\n", checksum);

  CUDA_CHECK(cudaEventDestroy(t_start));
  CUDA_CHECK(cudaEventDestroy(t_stop));
  CUDA_CHECK(cudaEventDestroy(ev_h2d_done));
  CUDA_CHECK(cudaEventDestroy(ev_a_done));
  CUDA_CHECK(cudaEventDestroy(ev_b_done));
  CUDA_CHECK(cudaStreamDestroy(stream_main));
  CUDA_CHECK(cudaStreamDestroy(stream_a));
  CUDA_CHECK(cudaStreamDestroy(stream_b));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
