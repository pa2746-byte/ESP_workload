#pragma once
// Analysis-only CUDA SDK declarations; never use these for a CUDA executable.
#include "cuda.h"
#include <math.h>
#define checkCudaErrors(expression) (expression)
void getLastCudaError(const char*);
