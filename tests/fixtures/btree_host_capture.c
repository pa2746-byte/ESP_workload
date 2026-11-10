#include "common.h"
#include <stdio.h>
#include <stdlib.h>
int cudaSetDevice(int n){return 0;}
int rand(void){return 0;}
void kernel_gpu_cuda_wrapper(record*r,long rm,knode*k,long ke,long km,int order,long h,int n,long*curr,long*off,int*keys,record*ans){
 FILE*f=fopen("captured_records.bin","wb");fwrite(r,1,rm,f);fclose(f);
 f=fopen("captured_knodes.bin","wb");fwrite(k,1,km,f);fclose(f);
 printf("CAPTURE %ld %ld %ld %d %ld %d %d\n",rm,ke,km,order,h,n,keys[0]);
}
void kernel_gpu_cuda_wrapper_2(void){}
