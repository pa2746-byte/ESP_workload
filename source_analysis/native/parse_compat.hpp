#pragma once
// macOS lacks POSIX barriers. This token may only stand in for an UNUSED
// serial-path barrier object; the native backend rejects pthread barrier calls.
#if defined(__APPLE__)
struct pthread_barrier_t { unsigned char unused_serial_token; };
#endif
using CUresult=int;
using CUdevice=int;
int cuInit(unsigned);
int cuDeviceGetCount(int*);
int cuDeviceComputeCapability(int*,int*,int);
int cuDeviceGet(int*,int);
int cuMemGetInfo(size_t*,size_t*);
enum cudaChannelFormatKind {cudaChannelFormatKindSigned,cudaChannelFormatKindUnsigned,cudaChannelFormatKindFloat};
cudaChannelFormatDesc cudaCreateChannelDesc(int,int,int,int,cudaChannelFormatKind);
int cudaMallocPitch(void**,size_t*,size_t,size_t);
int cudaMemcpy2D(void*,size_t,const void*,size_t,size_t,size_t,cudaMemcpyKind);
int cudaDeviceReset();
__device__ double atan(double);
struct StopWatchInterface;
float sdkGetTimerValue(StopWatchInterface**);
void sdkResetTimer(StopWatchInterface**);
__host__ __device__ float4 make_float4(float,float,float,float);

enum cudaTextureAddressMode {cudaAddressModeWrap,cudaAddressModeClamp,cudaAddressModeBorder};
template<class T,int D,cudaTextureReadMode R> int cudaBindTexture(size_t*,texture<T,D,R>&,const void*,cudaChannelFormatDesc,size_t);
