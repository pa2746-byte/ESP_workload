#pragma once
// Analysis-only declarations for Clang's host-side CUDA AST parser.
// Never put this include directory on the nvcc build path: this is not a CUDA runtime.
#include <stdlib.h>
#include <string.h>
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
struct uint3 { unsigned x,y,z; };
extern const uint3 threadIdx, blockIdx, blockDim, gridDim;
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
template<class T> int cudaMalloc(T**,size_t);
int cudaFree(void*); int cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind); int cudaDeviceSynchronize();
int cudaGetDeviceCount(int*); int cudaSetDevice(int);
int cudaConfigureCall(dim3,dim3,size_t=0,void* =nullptr); int cudaSetupArgument(const void*,size_t,size_t); int cudaLaunch(const void*);
// Clang selects this helper for the newer CUDA launch ABI.
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void* = nullptr);
__attribute__((device)) void __syncthreads();
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess = 0;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
enum cudaFuncCache { cudaFuncCachePreferL1 };
template<class T> cudaError_t cudaFuncSetCacheConfig(T, cudaFuncCache);
#define __constant__ __attribute__((constant))
struct float3 { float x, y, z; };
int cudaGetDevice(int*);
int cudaMemset(void*, int, size_t);
template<class T> int cudaMemcpyToSymbol(const T&, const void*, size_t, size_t = 0, cudaMemcpyKind = cudaMemcpyHostToDevice);

// Legacy CUDA runtime entry point, retained for original Rodinia sources.
int cudaThreadSynchronize();
__device__ double exp(double);

#include "cuda.h"

typedef int cudaError;

// Legacy texture declarations, analysis only.
enum cudaTextureReadMode {cudaReadModeElementType};
enum cudaTextureFilterMode {cudaFilterModePoint};
struct cudaChannelFormatDesc {};
template<class T,int D,cudaTextureReadMode R> struct texture { int filterMode; bool normalized; cudaChannelFormatDesc channelDesc; };
template<class T> cudaChannelFormatDesc cudaCreateChannelDesc();
template<class T,int D,cudaTextureReadMode R> int cudaBindTexture(size_t*,texture<T,D,R>*,const void*,const cudaChannelFormatDesc*,size_t);
template<class T,int D,cudaTextureReadMode R> __device__ T tex1Dfetch(texture<T,D,R>,int);
#define CUDA_SUCCESS 0

#include "vector_types.h"

// Declarations only: parsing these operations does not enable evaluator support.
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ inline __attribute__((always_inline))
template<class T> int cudaMallocHost(T**,size_t);
int cudaFreeHost(void*);
int cudaStreamCreate(cudaStream_t*);
int cudaEventCreate(cudaEvent_t*);
int cudaEventSynchronize(cudaEvent_t);
int cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);
int cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t=0);
template<class T> __device__ T atomicAdd(T*,T);
template<class T> __device__ T atomicAnd(T*,T);
template<class T> __device__ T atomicOr(T*,T);
__device__ unsigned atomicInc(unsigned*,unsigned);
__device__ double pow(double,double);
__device__ double fabs(double);
__device__ double abs(double);
__device__ int abs(int);
__device__ double fmod(double,double);
__device__ double log10(double);
__device__ double sin(double);
__device__ double cos(double);
__device__ double tanh(double);
template<class T,int D,cudaTextureReadMode R> int cudaBindTexture(size_t*,texture<T,D,R>&,const void*,size_t);
template<class T,int D,cudaTextureReadMode R> int cudaUnbindTexture(texture<T,D,R>&);
template<class T,int D,cudaTextureReadMode R> __device__ T tex2D(texture<T,D,R>,float,float);
__device__ unsigned atomicAdd(unsigned*,int);
__device__ int __mul24(int,int);
__device__ unsigned __umul24(unsigned,unsigned);
__device__ bool isnan(double);
template<class T> __host__ __device__ T max(T,T);
template<class T> __host__ __device__ T min(T,T);
__host__ __device__ int2 make_int2(int,int);
__host__ __device__ uchar3 make_uchar3(unsigned char,unsigned char,unsigned char);
#if defined(__APPLE__)
// Opaque declaration for signatures only. No pthread execution model is implied.
struct pthread_barrier_t;
#endif
__device__ bool isinf(double);
__host__ __device__ unsigned max(int,unsigned);
__host__ __device__ unsigned max(unsigned,int);
int cudaThreadExit();
// Legacy SDK helper signature, parsed as a host call and checked by evaluation.
#ifndef checkCudaErrors
int checkCudaErrors(int);
#endif

#ifndef __launch_bounds__
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#endif
