// CUPTI activity timestamps: no framework imports and no CUDA-event launch gaps.
#include <cupti.h>
#include <cupti_activity.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
struct Row {std::string kind,name;uint64_t start,end;uint32_t correlation,stream;uint32_t regs=0,local=0,shared=0;};
struct Stats {uint64_t count=0,ns=0;};
std::mutex mutex;
std::vector<Row> rows;
std::unordered_map<uint32_t,uint64_t> correlation;
std::unordered_map<uint32_t,Stats> apis;
std::string error;
size_t dropped=0;
bool initialized=false;
void check(CUptiResult status){if(status!=CUPTI_SUCCESS){const char* message=nullptr;cuptiGetResultString(status,&message);throw std::runtime_error(message?message:"CUPTI failure");}}
void CUPTIAPI request(uint8_t** buffer,size_t* size,size_t* maximum){*size=4*1024*1024;*maximum=0;*buffer=static_cast<uint8_t*>(std::malloc(*size));}
void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid){
    CUpti_Activity* record=nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    while(cuptiActivityGetNextRecord(buffer,valid,&record)==CUPTI_SUCCESS){
        if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
            auto* k=reinterpret_cast<CUpti_ActivityKernel11*>(record);
            rows.push_back({"kernel",k->name?k->name:"",k->start,k->end,k->correlationId,k->streamId,
                            k->registersPerThread,k->localMemoryPerThread,static_cast<uint32_t>(k->staticSharedMemory+k->dynamicSharedMemory)});
        }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
            auto* m=reinterpret_cast<CUpti_ActivityMemcpy6*>(record);
            rows.push_back({"memcpy",std::to_string(m->copyKind),m->start,m->end,m->correlationId,m->streamId});
        }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMSET){
            auto* m=reinterpret_cast<CUpti_ActivityMemset4*>(record);
            rows.push_back({"memset","memset",m->start,m->end,m->correlationId,m->streamId});
        }else if(record->kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION){
            auto* c=reinterpret_cast<CUpti_ActivityExternalCorrelation*>(record);
            if(c->externalKind==CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)correlation[c->correlationId]=c->externalId;
        }else if(record->kind==CUPTI_ACTIVITY_KIND_DRIVER || record->kind==CUPTI_ACTIVITY_KIND_RUNTIME){
            auto* a=reinterpret_cast<CUpti_ActivityAPI*>(record);
            auto& s=apis[a->cbid|(record->kind==CUPTI_ACTIVITY_KIND_RUNTIME?0x80000000u:0u)];s.count++;s.ns+=a->end-a->start;
        }
    }
    size_t lost=0;cuptiActivityGetNumDroppedRecords(context,stream,&lost);dropped+=lost;
    std::free(buffer);
}
const CUpti_ActivityKind kinds[]={CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET,CUPTI_ACTIVITY_KIND_DRIVER,CUPTI_ACTIVITY_KIND_RUNTIME,CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};
}
API const char* trace_error(){return error.c_str();}
API int trace_start(){try{
    if(!initialized){check(cuptiActivityRegisterCallbacks(request,complete));initialized=true;}
    {std::lock_guard<std::mutex> lock(mutex);rows.clear();correlation.clear();apis.clear();dropped=0;}
    for(auto kind:kinds)check(cuptiActivityEnable(kind));return 0;
}catch(const std::exception& e){error=e.what();return -1;}}
API int trace_push(uint64_t id){return static_cast<int>(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,id));}
API int trace_pop(){uint64_t id;return static_cast<int>(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&id));}
API int trace_stop(const char* path){try{
    check(cuptiActivityFlushAll(0));for(auto kind:kinds)check(cuptiActivityDisable(kind));
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot open trace output");
    for(const auto& r:rows){
        auto found=correlation.find(r.correlation);uint64_t label=found==correlation.end()?0:found->second;
        out<<"{\"kind\":"<<std::quoted(r.kind)<<",\"name\":"<<std::quoted(r.name)
           <<",\"start\":"<<r.start<<",\"end\":"<<r.end<<",\"label\":"<<label<<",\"stream\":"<<r.stream
           <<",\"registers\":"<<r.regs<<",\"local_bytes\":"<<r.local<<",\"shared_bytes\":"<<r.shared<<"}\n";
    }
    for(const auto& entry:apis){const char* name=nullptr;
        cuptiGetCallbackName((entry.first&0x80000000u)?CUPTI_CB_DOMAIN_RUNTIME_API:CUPTI_CB_DOMAIN_DRIVER_API,entry.first&0x7fffffffu,&name);
        out<<"{\"kind\":\"api\",\"name\":"<<std::quoted(name?name:"")<<",\"count\":"<<entry.second.count<<",\"ns\":"<<entry.second.ns<<"}\n";}
    out<<"{\"kind\":\"summary\",\"dropped\":"<<dropped<<",\"records\":"<<rows.size()<<"}\n";
    if(dropped)throw std::runtime_error("CUPTI dropped activity records");return 0;
}catch(const std::exception& e){error=e.what();return -1;}}
