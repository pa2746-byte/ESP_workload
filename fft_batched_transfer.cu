#include <cuda_runtime.h>
#include <cufft.h>
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

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult err__ = (call);                                                \
    if (err__ != CUFFT_SUCCESS) {                                              \
      std::fprintf(stderr, "cuFFT error %s:%d: code=%d\n", __FILE__, __LINE__,\
                   static_cast<int>(err__));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  // Args: fft_size batch iters mode
  // mode=0 => transfer once (H2D once + D2H once)
  // mode=1 => transfer per-iter (H2D + D2H each iteration)
  int fft_size = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int batch = (argc > 2) ? std::atoi(argv[2]) : 256;
  int iters = (argc > 3) ? std::atoi(argv[3]) : 50;
  int mode = (argc > 4) ? std::atoi(argv[4]) : 1;

  if (fft_size <= 0 || batch <= 0 || iters <= 0 || (mode != 0 && mode != 1)) {
    std::fprintf(stderr, "Usage: %s [fft_size] [batch] [iters] [mode 0|1]\n", argv[0]);
    return EXIT_FAILURE;
  }

  size_t elems = static_cast<size_t>(fft_size) * batch;
  size_t bytes = elems * sizeof(cufftComplex);

  cufftComplex *h_in = nullptr, *h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));

  for (size_t i = 0; i < elems; i++) {
    float x = static_cast<float>(i % fft_size);
    h_in[i].x = std::sin(2.0f * 3.1415926535f * x / fft_size);
    h_in[i].y = 0.0f;
  }

  cufftComplex* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cufftHandle plan;
  int n[1] = {fft_size};
  int inembed[1] = {fft_size};
  int onembed[1] = {fft_size};
  int istride = 1, ostride = 1;
  int idist = fft_size, odist = fft_size;

  CUFFT_CHECK(cufftPlanMany(&plan, 1, n,
                            inembed, istride, idist,
                            onembed, ostride, odist,
                            CUFFT_C2C, batch));
  CUFFT_CHECK(cufftSetStream(plan, stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  if (mode == 0) {
    nvtxRangePushA("H2D_once");
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_in, bytes, cudaMemcpyHostToDevice, stream));
    nvtxRangePop();
  }

  CUDA_CHECK(cudaEventRecord(start, stream));

  for (int it = 0; it < iters; it++) {
    if (mode == 1) {
      nvtxRangePushA("H2D_per_iter");
      CUDA_CHECK(cudaMemcpyAsync(d_data, h_in, bytes, cudaMemcpyHostToDevice, stream));
      nvtxRangePop();
    }

    nvtxRangePushA("cufftExecC2C_forward");
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    nvtxRangePop();

    if (mode == 1) {
      nvtxRangePushA("D2H_per_iter");
      CUDA_CHECK(cudaMemcpyAsync(h_out, d_data, bytes, cudaMemcpyDeviceToHost, stream));
      nvtxRangePop();
    }
  }

  if (mode == 0) {
    nvtxRangePushA("D2H_once");
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_data, bytes, cudaMemcpyDeviceToHost, stream));
    nvtxRangePop();
  }

  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double mag_sum = 0.0;
  for (size_t i = 0; i < 1024 && i < elems; i++) {
    mag_sum += std::sqrt(double(h_out[i].x) * h_out[i].x + double(h_out[i].y) * h_out[i].y);
  }

  double expectedTransferBytes =
      (mode == 0) ? (2.0 * bytes) : (iters * 2.0 * bytes);

  std::printf("Batched FFT complete\n");
  std::printf("fft_size=%d, batch=%d, iters=%d, mode=%d (%s)\n",
              fft_size, batch, iters, mode,
              mode == 0 ? "transfer_once" : "transfer_per_iter");
  std::printf("Elapsed: %.3f ms\n", ms);
  std::printf("Expected host-device transfer bytes: %.0f\n", expectedTransferBytes);
  std::printf("Magnitude sum(0:1024): %.6f\n", mag_sum);

  CUFFT_CHECK(cufftDestroy(plan));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
