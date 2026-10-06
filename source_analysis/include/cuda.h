#pragma once
// Analysis-only compatibility for Rodinia sources relying on nvcc's implicit headers.
#include "cuda_runtime.h"
#include <string.h>
struct cudaDeviceProp { int maxGridSize[3]; int maxThreadsPerBlock; };
int cudaGetDeviceProperties(cudaDeviceProp*, int);
int cudaMemGetInfo(size_t*, size_t*);
int cudaThreadSynchronize();
__device__ double sqrt(double);
