#pragma once
// Analysis-only compatibility for Rodinia sources relying on nvcc's implicit headers.
#include "cuda_runtime.h"
#include <string.h>
struct cudaDeviceProp {
    int maxGridSize[3], maxThreadsPerBlock, maxThreadsDim[3];
    char name[256];
    size_t totalGlobalMem, sharedMemPerBlock, memPitch, totalConstMem, textureAlignment;
    int regsPerBlock, warpSize, major, minor, clockRate, deviceOverlap, multiProcessorCount;
};
int cudaGetDeviceCount(int*);
int cudaGetDeviceProperties(cudaDeviceProp*, int);
int cudaMemGetInfo(size_t*, size_t*);
int cudaThreadSynchronize();
__device__ double sqrt(double);
__device__ float __log2f(float);
__device__ float __powf(float, float);
__device__ double exp(double);
__device__ double log(double);
__device__ float sqrtf(float);
