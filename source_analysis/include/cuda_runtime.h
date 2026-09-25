#pragma once
// Analysis-only declarations for Clang's host-side CUDA AST parser.
// Never put this include directory on the nvcc build path: this is not a CUDA runtime.
#include <stdlib.h>
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
