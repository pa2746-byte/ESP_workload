#pragma once
// Analysis-only declarations for Clang's host-side CUDA AST parser.
// Never put this include directory on the nvcc build path: this is not a CUDA runtime.
#include <stdlib.h>
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
struct uint3 { unsigned x,y,z; };
extern const uint3 threadIdx, blockIdx, blockDim, gridDim;
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
template<class T> int cudaMalloc(T**,size_t);
int cudaFree(void*); int cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind); int cudaDeviceSynchronize();
int cudaConfigureCall(dim3,dim3,size_t=0,void* =nullptr); int cudaSetupArgument(const void*,size_t,size_t); int cudaLaunch(const void*);
__attribute__((device)) void __syncthreads();
