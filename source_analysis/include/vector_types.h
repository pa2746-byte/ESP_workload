#pragma once
// Analysis-only CUDA vector declarations.
#include "cuda_runtime.h"
struct float2 { float x,y; };
struct float4 { float x,y,z,w; };
struct int2 { int x,y; };
struct int4 { int x,y,z,w; };
struct uint2 { unsigned x,y; };
struct uint4 { unsigned x,y,z,w; };
struct uchar3 { unsigned char x,y,z; };
struct uchar4 { unsigned char x,y,z,w; };
