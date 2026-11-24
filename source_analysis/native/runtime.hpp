#pragma once
// CPU source semantics only. This runtime does not call CUDA or a GPU profiler.
#include <vector>
#include <map>
#include <set>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include "block_executor.hpp"

inline thread_local uint3 threadIdx{}, blockIdx{}, blockDim{}, gridDim{};
namespace esp_source {
struct Allocation {
    int id; unsigned char* data; size_t size; bool owned;
    std::vector<unsigned char> initialized;
    std::set<size_t> reads,writes,inputs;
    std::map<size_t,History> writer;
    std::map<size_t,std::map<int,History>> readers;
    bool constant=false;unsigned texture_bindings=0;bool texture_active=false;
};
inline std::map<int,Allocation> allocations;
inline int next_id=0;
inline thread_local int lane=0;
inline thread_local bool kernel_active=false;
inline thread_local uintptr_t stack_top=0;
inline std::map<uintptr_t,Allocation> shared_allocations;
inline std::vector<unsigned char> dynamic_shared;
inline std::ofstream output;
inline void fail(const std::string& reason){throw std::runtime_error("SOURCE_ANALYSIS: "+reason);}
inline std::ofstream& log(){
    if(!output.is_open()){
        auto path=std::getenv("ESP_SOURCE_EVENTS");
        if(!path)fail("ESP_SOURCE_EVENTS is required");
        output.open(path);output.setf(std::ios::unitbuf);if(!output)fail("Cannot open source event file");
    }
    return output;
}
inline std::string quote(const std::string& s){
    std::string out="\"";
    for(char c:s){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else out+=c;}
    return out+'"';
}
template<class Map> inline Allocation* locate(Map& storage,const void* address,size_t size,bool endpoint=false){
    auto p=(uintptr_t)address;Allocation* end=nullptr;
    for(auto& entry:storage){auto& a=entry.second;auto begin=(uintptr_t)a.data;
        if(p>=begin&&p<begin+a.size){if(size>a.size||p-begin>a.size-size)fail("Access exceeds device/shared allocation");return &a;}
        if(endpoint&&p==begin+a.size)end=&a;
    }
    if(end&&size)fail("Access exceeds device/shared allocation");
    return end;
}
inline Allocation* find(const void* address,size_t size,bool endpoint=false){return locate(allocations,address,size,endpoint);}
inline Allocation* find_shared(const void* address,size_t size,bool endpoint=false){return locate(shared_allocations,address,size,endpoint);}

inline int register_allocation(void* pointer,size_t size,bool owned){
    if(!pointer||!size||size>16*1024*1024)fail("Invalid or excessive source allocation");
    int id=next_id++;
    Allocation a{id,(unsigned char*)pointer,size,owned,std::vector<unsigned char>(size,0)};
    allocations.emplace(id,std::move(a));
    log()<<"{\"event\":\"allocate\",\"buffer\":"<<id<<",\"size\":"<<size<<"}\n";
    return id;
}
inline void ranges(std::ostream& o,const std::set<size_t>& offsets){
    o<<'[';bool first=true;auto i=offsets.begin();
    while(i!=offsets.end()){size_t start=*i,last=start;++i;while(i!=offsets.end()&&*i==last+1){last=*i;++i;}
        if(!first)o<<',';first=false;o<<'['<<start<<','<<last+1<<']';}
    o<<']';
}
inline void touch(const void* address,size_t size,int mode,bool atomic=false){
    if(!kernel_active)return;
    std::lock_guard<std::recursive_mutex> guard(memory_mutex);
    if(block_sync)block_sync->check();
    Allocation* a=find(address,size,true);bool shared=false;
    if(!a){a=find_shared(address,size,true);shared=a!=nullptr;}
    if(!a)return;
    if(shared&&shared_mode()!="checked_blocks")fail("Executed shared-memory access is unsupported in this mode");
    size_t start=(uintptr_t)address-(uintptr_t)a->data;
    bool read_only=a->constant||a->texture_active;
    if(read_only&&(mode&2))fail("Kernel writes constant or texture-bound storage");
    History current{lane,current_block,current_phase,atomic};
    for(size_t i=start;i<start+size;++i){
        auto writer=a->writer.find(i);
        if(mode&1){
            if(!a->initialized[i])fail("Kernel reads uninitialized device/shared bytes");
            if(writer!=a->writer.end()&&!ordered(writer->second,lane,atomic))fail("Cross-thread read/write conflict without an ordering barrier");
            a->reads.insert(i);if(writer==a->writer.end())a->inputs.insert(i);
            if(!read_only){auto prior=a->readers[i].find(lane);History read=current;
                if(prior!=a->readers[i].end()&&prior->second.phase==current_phase)read.atomic=atomic&&prior->second.atomic;
                a->readers[i][lane]=read;}
        }
        if(mode&2){
            if(writer!=a->writer.end()&&!ordered(writer->second,lane,atomic))fail("Multiple threads write the same bytes without an ordering barrier");
            for(auto& read:a->readers[i])if(!ordered(read.second,lane,atomic))fail("Cross-thread write/read conflict without an ordering barrier");
            History write=current;
            if(writer!=a->writer.end()&&writer->second.lane==lane&&writer->second.phase==current_phase)write.atomic=atomic&&writer->second.atomic;
            a->writer[i]=write;a->writes.insert(i);a->initialized[i]=1;
        }
    }
}
inline void check_pointer(const void* pointer,size_t size){
    if(!kernel_active)return;
    std::lock_guard<std::recursive_mutex> guard(memory_mutex);
    if(block_sync)block_sync->check();
    if(find(pointer,size,true)||find_shared(pointer,size,true))return;
    auto p=(uintptr_t)pointer;
    if(p>stack_top-8*1024*1024&&p<stack_top)return;
    fail("Kernel dereferences unmodeled host/static storage");
}

inline void copy(void* destination,const void* source,size_t size,cudaMemcpyKind kind){
    if(!size)return;
    auto dst=find(destination,size,true),src=find(source,size,true);
    if(kind==cudaMemcpyHostToDevice && (!dst||src))fail("Invalid host-to-device copy");
    if(kind==cudaMemcpyDeviceToHost && (!src||dst))fail("Invalid device-to-host copy");
    if(kind==cudaMemcpyDeviceToDevice && (!src||!dst))fail("Invalid device-to-device copy");
    if(src){size_t offset=(uintptr_t)source-(uintptr_t)src->data;
        for(size_t i=0;i<size;++i)if(!src->initialized[offset+i])fail("Copy reads uninitialized device bytes");}
    std::memmove(destination,source,size);
    if(dst){size_t offset=(uintptr_t)destination-(uintptr_t)dst->data;for(size_t i=0;i<size;++i)dst->initialized[offset+i]=1;}
    log()<<"{\"event\":\"copy\",\"size\":"<<size<<",\"source\":"<<(src?src->id:-1)
         <<",\"destination\":"<<(dst?dst->id:-1)<<",\"source_offset\":"<<(src?(uintptr_t)source-(uintptr_t)src->data:0)
         <<",\"destination_offset\":"<<(dst?(uintptr_t)destination-(uintptr_t)dst->data:0)<<"}\n";
}
}

template<class T> T esp_source_argument(T value){return value;}
template<class T> T* esp_source_argument(T* pointer){
    if(pointer && !esp_source::find(pointer,0,true))esp_source::fail("Kernel argument is not device storage");
    return pointer;
}
template<class T> T& esp_source_access(T& value,int mode){esp_source::touch((const void*)&value,sizeof(T),mode);return value;}
template<class T,class I> T& esp_source_index(T* pointer,I index){
    // ASan/UBSan independently check native heap/stack bounds as well.
    T* target=pointer+index;esp_source::check_pointer(target,sizeof(T));return *target;
}
template<class T> T* esp_source_pointer(T* pointer){esp_source::check_pointer(pointer,sizeof(T));return pointer;}
inline void esp_source_shared(void* pointer,size_t size,const char*){
    using namespace esp_source;
    std::lock_guard<std::recursive_mutex> guard(memory_mutex);
    if(!kernel_active||shared_mode()=="reject")fail("Shared declaration is unsupported in this mode");
    auto key=(uintptr_t)pointer;auto it=shared_allocations.find(key);
    if(it!=shared_allocations.end()){if(it->second.size!=size)fail("Shared aliases have inconsistent extents");return;}
    if(!size||size>49152)fail("Shared allocation exceeds configured source limit");
    shared_allocations.emplace(key,Allocation{-1,(unsigned char*)pointer,size,false,std::vector<unsigned char>(size,0)});
}
inline void* esp_source_dynamic_shared(const char* name){
    if(esp_source::dynamic_shared.empty())esp_source::fail("Dynamic shared memory has zero launch extent");
    esp_source_shared(esp_source::dynamic_shared.data(),esp_source::dynamic_shared.size(),name);
    return esp_source::dynamic_shared.data();
}
inline void esp_source_barrier(const char* site){
    if(esp_source::shared_mode()!="checked_blocks"||!esp_source::block_sync)esp_source::fail("Executed block barrier is unsupported in this mode");
    esp_source::block_sync->wait(site);++esp_source::current_phase;
}
template<class T,class F> inline T esp_source_atomic(T* pointer,F update){
    using namespace esp_source;
    if(shared_mode()!="checked_blocks")fail("Atomics require checked block execution");
    std::lock_guard<std::recursive_mutex> guard(memory_mutex);
    check_pointer(pointer,sizeof(T));touch(pointer,sizeof(T),3,true);T old=*pointer;*pointer=update(old);return old;
}
template<class T> inline T atomicAdd(T* pointer,T value){return esp_source_atomic(pointer,[&](T old){return old+value;});}
template<class T> inline T atomicAnd(T* pointer,T value){return esp_source_atomic(pointer,[&](T old){return old&value;});}
template<class T> inline T atomicOr(T* pointer,T value){return esp_source_atomic(pointer,[&](T old){return old|value;});}
inline unsigned atomicAdd(unsigned* pointer,int value){return atomicAdd(pointer,(unsigned)value);}
inline unsigned atomicInc(unsigned* pointer,unsigned limit){return esp_source_atomic(pointer,[&](unsigned old){return old>=limit?0u:old+1;});}
template<class G,class B,class F,class S> void esp_source_launch(G raw_grid,B raw_block,size_t shared,const char* name,F function,S stream){
    using namespace esp_source;
    if(stream!=0&&(!std::getenv("ESP_SOURCE_SERIAL_STREAMS")||std::string(std::getenv("ESP_SOURCE_SERIAL_STREAMS"))!="1"))fail("Native source execution requires the default stream");
    auto mode=shared_mode();if(shared&&mode=="reject")fail("Dynamic shared memory is unsupported in this mode");
    if(shared>49152)fail("Dynamic shared memory exceeds source limit");
    dim3 grid(raw_grid),block(raw_block);size_t per_block=(size_t)block.x*block.y*block.z;
    size_t threads=(size_t)grid.x*grid.y*grid.z*per_block;
    if(!threads||threads>262144||per_block>1024)fail("Launch exceeds bounded source limits");
    for(auto& item:allocations){item.second.texture_active=false;item.second.reads.clear();item.second.inputs.clear();item.second.writes.clear();item.second.writer.clear();item.second.readers.clear();}
    int block_number=0;unsigned barriers=0;
    for(unsigned z=0;z<grid.z;++z)for(unsigned y=0;y<grid.y;++y)for(unsigned x=0;x<grid.x;++x,++block_number){
        shared_allocations.clear();dynamic_shared.assign(shared,0);
        auto execute=[&](unsigned index,BlockSync* sync){
            block_sync=sync;current_block=block_number;current_phase=0;
            gridDim={grid.x,grid.y,grid.z};blockDim={block.x,block.y,block.z};blockIdx={x,y,z};
            threadIdx={index%block.x,(index/block.x)%block.y,index/(block.x*block.y)};
            kernel_active=true;lane=block_number*per_block+index;stack_top=(uintptr_t)__builtin_frame_address(0);
            try{function();}catch(...){kernel_active=false;block_sync=nullptr;throw;}
            kernel_active=false;block_sync=nullptr;
        };
        if(mode=="checked_blocks"){
            BlockSync sync((unsigned)per_block);std::vector<std::thread> workers;
            try{
                for(unsigned index=0;index<per_block;++index)workers.emplace_back([&,index]{
                    try{execute(index,&sync);}catch(...){sync.cancel(std::current_exception());}sync.done();
                });
            }catch(...){sync.cancel(std::current_exception());}
            for(auto& worker:workers)worker.join();sync.check();barriers+=sync.generation;
        }else for(unsigned index=0;index<per_block;++index)execute(index,nullptr);
    }
    shared_allocations.clear();dynamic_shared.clear();kernel_active=false;
    auto& out=log();out<<"{\"event\":\"kernel\",\"name\":"<<quote(name)<<",\"grid\":["<<grid.x<<','<<grid.y<<','<<grid.z
        <<"],\"block\":["<<block.x<<','<<block.y<<','<<block.z<<"],\"checked_block_barriers\":"<<barriers<<",\"accesses\":[";
    bool first=true;for(auto& item:allocations){auto& a=item.second;if(a.reads.empty()&&a.writes.empty())continue;
        if(!first)out<<',';first=false;out<<"{\"buffer\":"<<a.id<<",\"reads\":";ranges(out,a.reads);out<<",\"external_reads\":";ranges(out,a.inputs);out<<",\"writes\":";ranges(out,a.writes);out<<'}';}
    out<<"]}\n";out.flush();
}

template<class T> inline int cudaMalloc(T** pointer,size_t size){
    if(!size)esp_source::fail("Zero device allocation");
    *pointer=(T*)std::malloc(size);esp_source::register_allocation(*pointer,size,true);return 0;
}
inline int cudaFree(void* pointer){
    if(!pointer)return 0;auto a=esp_source::find(pointer,0);
    if(!a||pointer!=a->data||!a->owned)esp_source::fail("Invalid device free");
    int id=a->id;std::free(a->data);esp_source::allocations.erase(id);
    esp_source::log()<<"{\"event\":\"free\",\"buffer\":"<<id<<"}\n";return 0;
}
inline int cudaMemcpy(void* dst,const void* src,size_t size,cudaMemcpyKind kind){esp_source::copy(dst,src,size,kind);return 0;}
inline int cudaMemset(void* pointer,int value,size_t size){
    auto a=esp_source::find(pointer,size,true);if(!a)esp_source::fail("Invalid device memset");
    size_t offset=(uintptr_t)pointer-(uintptr_t)a->data;std::memset(pointer,value,size);
    for(size_t i=offset;i<offset+size;++i)a->initialized[i]=1;
    esp_source::log()<<"{\"event\":\"memset\",\"buffer\":"<<a->id<<",\"offset\":"<<offset<<",\"size\":"<<size<<"}\n";return 0;
}
template<class T> inline int cudaMemcpyToSymbol(const T& symbol,const void* source,size_t size,size_t offset,cudaMemcpyKind kind){
    if(offset+size>sizeof(T)||kind!=cudaMemcpyHostToDevice)esp_source::fail("Unsupported constant copy");
    if(!esp_source::find(&symbol,sizeof(T)))esp_source::register_allocation((void*)&symbol,sizeof(T),false);
    esp_source::find(&symbol,sizeof(T))->constant=true;
    return cudaMemcpy((char*)&symbol+offset,source,size,kind);
}
template<class T> inline int cudaMallocHost(T** pointer,size_t size){*pointer=(T*)std::malloc(size);return *pointer?0:1;}
inline int cudaFreeHost(void* pointer){std::free(pointer);return 0;}
inline int cudaThreadSynchronize(){return 0;}
inline int cudaDeviceSynchronize(){return 0;}
inline int cudaThreadExit(){return 0;}
inline int cudaGetLastError(){return 0;}
inline const char* cudaGetErrorString(int error){return error?"CPU source runtime error":"CPU source runtime success";}
inline int cudaGetDeviceCount(int* count){*count=1;return 0;}
inline int cudaGetDevice(int* id){*id=0;return 0;}
inline int cudaSetDevice(int id){if(id)esp_source::fail("Unconfigured source device");return 0;}
inline int cudaGetDeviceProperties(cudaDeviceProp* p,int id){
    if(id)esp_source::fail("Unconfigured source device");std::memset(p,0,sizeof(*p));
    p->major=7;p->minor=5;p->multiProcessorCount=1;p->warpSize=32;p->deviceOverlap=std::getenv("ESP_SOURCE_SERIAL_STREAMS")&&std::string(std::getenv("ESP_SOURCE_SERIAL_STREAMS"))=="1";p->maxThreadsPerBlock=1024;p->maxThreadsDim[0]=1024;p->maxThreadsDim[1]=1024;p->maxThreadsDim[2]=64;
    p->maxGridSize[0]=2147483647;p->maxGridSize[1]=65535;p->maxGridSize[2]=65535;p->totalGlobalMem=1ull<<30;p->sharedMemPerBlock=49152;
    std::strcpy(p->name,"Configured CPU source execution");return 0;
}
inline int cudaMemGetInfo(size_t* free_bytes,size_t* total){*free_bytes=*total=1ull<<30;return 0;}
inline int cudaEventCreate(cudaEvent_t* event){*event=(void*)1;return 0;}
inline int cudaEventRecord(cudaEvent_t,cudaStream_t){return 0;}
inline int cudaEventSynchronize(cudaEvent_t){return 0;}
inline int cudaEventDestroy(cudaEvent_t){return 0;}
inline int cudaEventElapsedTime(float* time,cudaEvent_t,cudaEvent_t){*time=0;return 0;}
inline int cuInit(unsigned){return 0;}
inline int cuDeviceGetCount(int* count){return cudaGetDeviceCount(count);}
inline int cuDeviceComputeCapability(int* major,int* minor,int id){if(id)esp_source::fail("Unconfigured device");*major=7;*minor=5;return 0;}
inline int cuDeviceGet(int* device,int id){if(id)esp_source::fail("Unconfigured device");*device=0;return 0;}
inline int cuMemGetInfo(size_t* a,size_t* b){return cudaMemGetInfo(a,b);}
inline cudaChannelFormatDesc cudaCreateChannelDesc(int,int,int,int,cudaChannelFormatKind){return {};}
inline int cudaMallocPitch(void** pointer,size_t* pitch,size_t width,size_t height){
    if(!height||width>16*1024*1024/height)esp_source::fail("Excessive pitched allocation");
    *pitch=width;return cudaMalloc(pointer,width*height);
}
inline int cudaMemcpy2D(void* destination,size_t dpitch,const void* source,size_t spitch,size_t width,size_t height,cudaMemcpyKind kind){
    if(width>dpitch||width>spitch)esp_source::fail("Invalid 2D copy pitch");
    for(size_t row=0;row<height;++row)cudaMemcpy((char*)destination+row*dpitch,(const char*)source+row*spitch,width,kind);
    return 0;
}
inline int cudaDeviceReset(){return 0;}
inline unsigned __umul24(unsigned a,unsigned b){return (a&0xffffffu)*(b&0xffffffu);}
inline int __mul24(int a,int b){
    int32_t x=(int32_t)((uint32_t)a<<8)>>8,y=(int32_t)((uint32_t)b<<8)>>8;
    return (int32_t)((int64_t)x*y);
}
inline int2 make_int2(int x,int y){return {x,y};}
inline int omp_get_num_threads(){return 1;}
inline int omp_get_thread_num(){return 0;}
inline int omp_get_num_procs(){return 1;}
inline void omp_set_num_threads(int count){if(count!=1)esp_source::fail("Native source execution requires serial OpenMP");}
inline uchar3 make_uchar3(unsigned char x,unsigned char y,unsigned char z){return {x,y,z};}
template<class T> inline cudaChannelFormatDesc cudaCreateChannelDesc(){return {};}
template<class T,int D,cudaTextureReadMode R> inline int cudaBindTexture(size_t* offset,texture<T,D,R>* tex,const void* pointer,const cudaChannelFormatDesc*,size_t bytes){
    auto allocation=esp_source::find(pointer,bytes,true);
    if(!allocation||pointer!=allocation->data||D!=1||bytes%sizeof(T))esp_source::fail("Unsupported or unaligned texture binding");
    if(tex->esp_data){auto old=esp_source::find(tex->esp_data,0);if(old&&old->texture_bindings)--old->texture_bindings;}
    ++allocation->texture_bindings;
    if(offset)*offset=0;tex->esp_data=(const T*)pointer;tex->esp_count=bytes/sizeof(T);return 0;
}
template<class T,int D,cudaTextureReadMode R> inline int cudaBindTexture(size_t* offset,texture<T,D,R>& tex,const void* pointer,size_t bytes){return cudaBindTexture(offset,&tex,pointer,(const cudaChannelFormatDesc*)nullptr,bytes);}
template<class T,int D,cudaTextureReadMode R> inline int cudaUnbindTexture(texture<T,D,R>& tex){if(tex.esp_data){auto a=esp_source::find(tex.esp_data,0);if(a&&a->texture_bindings)--a->texture_bindings;}tex.esp_data=nullptr;tex.esp_count=0;return 0;}
template<class T,int D,cudaTextureReadMode R> inline T tex1Dfetch(texture<T,D,R> tex,int index){
    if(D!=1||!tex.esp_data||index<0||(size_t)index>=tex.esp_count)esp_source::fail("Texture fetch outside modeled binding");
    std::lock_guard<std::recursive_mutex> guard(esp_source::memory_mutex);
    auto bound=esp_source::find(tex.esp_data,0);if(!bound||!bound->writes.empty())esp_source::fail("Kernel mixes texture reads with writes to the same storage");bound->texture_active=true;
    auto pointer=tex.esp_data+index;esp_source::check_pointer(pointer,sizeof(T));esp_source::touch(pointer,sizeof(T),1);return *pointer;
}
struct StopWatchInterface;
inline void sdkCreateTimer(StopWatchInterface** timer){*timer=nullptr;}
inline void sdkStartTimer(StopWatchInterface**){}
inline void sdkStopTimer(StopWatchInterface**){}
inline float sdkGetAverageTimerValue(StopWatchInterface**){return 0;}
inline float sdkGetTimerValue(StopWatchInterface**){return 0;}
inline void sdkResetTimer(StopWatchInterface**){}
inline void getLastCudaError(const char*){}
inline float4 make_float4(float a,float b,float c,float d){return {a,b,c,d};}

inline void esp_source_require_serial_streams(){if(!std::getenv("ESP_SOURCE_SERIAL_STREAMS")||std::string(std::getenv("ESP_SOURCE_SERIAL_STREAMS"))!="1")esp_source::fail("Nondefault streams require explicit serial source projection");}
inline int cudaStreamCreate(cudaStream_t* stream){esp_source_require_serial_streams();static uintptr_t next=1;*stream=(void*)next++;return 0;}
inline int cudaStreamSynchronize(cudaStream_t stream){if(stream)esp_source_require_serial_streams();return 0;}
inline int cudaStreamDestroy(cudaStream_t stream){if(stream)esp_source_require_serial_streams();return 0;}
inline int cudaMemcpyAsync(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream){if(stream)esp_source_require_serial_streams();return cudaMemcpy(dst,src,bytes,kind);}
template<class T,int D,cudaTextureReadMode R> inline int cudaBindTexture(size_t* offset,texture<T,D,R>& tex,const void* ptr,cudaChannelFormatDesc desc,size_t bytes){return cudaBindTexture(offset,&tex,ptr,&desc,bytes);}

inline unsigned max(int a,unsigned b){return a>b?(unsigned)a:b;}
inline unsigned max(unsigned a,int b){return a>(unsigned)b?a:(unsigned)b;}
