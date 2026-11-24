#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <exception>

namespace esp_source {
struct BlockSync {
    unsigned total,arrived=0,finished=0,generation=0;
    std::string site;
    std::mutex mutex;
    std::condition_variable condition;
    std::exception_ptr error;
    explicit BlockSync(unsigned count):total(count){}
    void cancel(std::exception_ptr failure){
        std::lock_guard<std::mutex> guard(mutex);if(!error)error=failure;condition.notify_all();
    }
    void check(){std::lock_guard<std::mutex> guard(mutex);if(error)std::rethrow_exception(error);}
    void wait(const char* location){
        std::unique_lock<std::mutex> guard(mutex);
        if(error)std::rethrow_exception(error);
        if(finished)throw std::runtime_error("SOURCE_ANALYSIS: Nonuniform block barrier participation");
        unsigned before=generation;
        if(!arrived)site=location;
        if(site!=location)throw std::runtime_error("SOURCE_ANALYSIS: Threads reach different block barrier sites");
        if(++arrived==total){arrived=0;++generation;condition.notify_all();return;}
        if(!condition.wait_for(guard,std::chrono::seconds(10),[&]{return error||generation!=before||finished;}))
            throw std::runtime_error("SOURCE_ANALYSIS: Block barrier timed out");
        if(error)std::rethrow_exception(error);
        if(generation==before)throw std::runtime_error("SOURCE_ANALYSIS: Nonuniform block barrier participation");
    }
    void done(){
        std::lock_guard<std::mutex> guard(mutex);++finished;
        if(arrived&&!error)error=std::make_exception_ptr(std::runtime_error("SOURCE_ANALYSIS: Nonuniform block barrier participation"));
        condition.notify_all();
    }
};
inline thread_local BlockSync* block_sync=nullptr;
inline thread_local int current_block=0,current_phase=0;
inline std::recursive_mutex memory_mutex;
inline std::string shared_mode(){const char* mode=std::getenv("ESP_SOURCE_SHARED");return mode?mode:"reject";}
struct History {int lane,block,phase;bool atomic;};
inline bool ordered(const History& prior,int current_lane,bool atomic){
    return prior.lane==current_lane || (prior.block==current_block&&prior.phase<current_phase) || (prior.atomic&&atomic);
}
}
