// Optional packed attention bridge. CUDA Runtime is linked explicitly.
#include <cudnn_frontend.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace fe=cudnn_frontend;
namespace {
thread_local std::string error;
void check(cudnnStatus_t s) { if(s!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(cudnnGetErrorString(s)); }
void check(fe::error_t s) { if(!s.is_good())throw std::runtime_error(s.get_message()); }
struct Plan {
    cudnnHandle_t handle=nullptr;
    fe::graph::Graph graph;
    bool backward=false;
    bool ragged=false;
    std::vector<int64_t> indices;
    int64_t selected=-1;
    ~Plan(){if(handle)cudnnDestroy(handle);}
};
}
API const char* sdpa_error(){return error.c_str();}
API int sdpa_create_ex(int64_t b,int64_t s,int64_t h,int64_t hk,int64_t d,int backward,int64_t total,void* stream,void** result,uint64_t* workspace,int tune,uint64_t limit){
  try {
    auto p=std::make_unique<Plan>();p->backward=backward!=0;p->ragged=total>0;
    check(cudnnCreate(&p->handle));check(cudnnSetStream(p->handle,static_cast<cudaStream_t>(stream)));
    auto& g=p->graph;
    g.set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto tensor=[&](int id,int64_t heads){return g.tensor(fe::graph::Tensor_attributes().set_uid(id).set_dim({b,heads,s,d}).set_stride({s*heads*d,d,heads*d,1}));};
    auto Q=tensor(1,h),K=tensor(2,hk),V=tensor(3,hk);
    std::shared_ptr<fe::graph::Tensor_attributes> qo,ko,lo,lens;
    if(total>0){
      auto offset=[&](int id){return g.tensor(fe::graph::Tensor_attributes().set_uid(id).set_dim({b+1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT64));};
      qo=offset(10);ko=offset(11);lo=offset(12);
      lens=g.tensor(fe::graph::Tensor_attributes().set_uid(13).set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));
      Q->set_ragged_offset(qo);K->set_ragged_offset(ko);V->set_ragged_offset(ko);
    }
    if(backward){
      auto O=tensor(4,h),DY=tensor(6,h);
      auto L=g.tensor(fe::graph::Tensor_attributes().set_uid(5).set_data_type(fe::DataType_t::FLOAT).set_dim({b,h,s,1}).set_stride({h*s,s,1,1}));
      auto options=fe::graph::SDPA_backward_attributes().set_attn_scale(1.f/std::sqrt(float(d))).set_causal_mask(true);
      if(total>0){
        O->set_ragged_offset(qo);DY->set_ragged_offset(qo);
        options.set_padding_mask(true).set_seq_len_q(lens).set_seq_len_kv(lens).set_max_total_seq_len_q(total).set_max_total_seq_len_kv(total);
      }
      auto [DQ,DK,DV]=g.sdpa_backward(Q,K,V,O,DY,L,options);
      DQ->set_output(true).set_uid(7).set_dim({b,h,s,d}).set_stride({s*h*d,d,h*d,1});
      DK->set_output(true).set_uid(8).set_dim({b,hk,s,d}).set_stride({s*hk*d,d,hk*d,1});
      DV->set_output(true).set_uid(9).set_dim({b,hk,s,d}).set_stride({s*hk*d,d,hk*d,1});
      if(total>0){DQ->set_ragged_offset(qo);DK->set_ragged_offset(ko);DV->set_ragged_offset(ko);}
    }else{
      auto options=fe::graph::SDPA_attributes().set_attn_scale(1.f/std::sqrt(float(d))).set_generate_stats(true).set_causal_mask(true);
      if(total>0)options.set_padding_mask(true).set_seq_len_q(lens).set_seq_len_kv(lens).set_max_total_seq_len_q(total).set_max_total_seq_len_kv(total);
      auto [O,L]=g.sdpa(Q,K,V,options);
      O->set_output(true).set_uid(4).set_dim({b,h,s,d}).set_stride({s*h*d,d,h*d,1});
      L->set_output(true).set_uid(5).set_data_type(fe::DataType_t::FLOAT).set_dim({b,h,s,1}).set_stride({h*s,s,1,1});
      if(total>0)O->set_ragged_offset(qo);
    }
    if(tune){
      g.deselect_workspace_greater_than(static_cast<int64_t>(limit));
      check(g.build(p->handle,{fe::HeurMode_t::A,fe::HeurMode_t::B,fe::HeurMode_t::FALLBACK},fe::BuildPlanPolicy_t::ALL));
    }else check(g.build(p->handle,{fe::HeurMode_t::A}));
    *workspace=g.get_workspace_size();
    std::string initial;check(g.get_plan_name(initial));
    for(int64_t i=0;i<g.get_execution_plan_count();++i){
      int64_t bytes=0;std::string name;
      if(!g.get_workspace_size_plan_at_index(i,bytes).is_good() || !g.get_plan_name_at_index(i,name).is_good())continue;
      if(tune && static_cast<uint64_t>(bytes)>limit)continue;
      if(name==initial && p->selected<0)p->selected=i;
      p->indices.push_back(i);
      if(tune)*workspace=std::max(*workspace,static_cast<uint64_t>(bytes));
    }
    *result=p.release();return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
API int sdpa_create(int64_t b,int64_t s,int64_t h,int64_t hk,int64_t d,int backward,int64_t total,void* stream,void** result,uint64_t* workspace){
  return sdpa_create_ex(b,s,h,hk,d,backward,total,stream,result,workspace,0,0);
}
API int sdpa_candidate_count(void* opaque){return static_cast<int>(static_cast<Plan*>(opaque)->indices.size());}
API int sdpa_candidate(void* opaque,int ordinal,uint64_t* workspace,const char** name){
  try{
    auto& p=*static_cast<Plan*>(opaque);auto index=p.indices.at(ordinal);
    static thread_local std::string tag;check(p.graph.get_plan_name_at_index(index,tag));
    *name=tag.c_str();*workspace=p.graph.get_workspace_size_plan_at_index(index);return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
API int sdpa_select(void* opaque,int ordinal){
  try{auto& p=*static_cast<Plan*>(opaque);p.selected=p.indices.at(ordinal);return 0;}
  catch(const std::exception& e){error=e.what();return -1;}
}
API int sdpa_run(void* opaque,void** buffers,void* workspace){
  try{
    auto& p=*static_cast<Plan*>(opaque);std::unordered_map<int64_t,void*> bindings;
    for(int i=1;i<=(p.backward?9:5);++i)bindings.emplace(i,buffers[i-1]);
    if(p.ragged)for(int i:{10,11,13})bindings.emplace(i,buffers[i-1]);
    if(p.selected>=0)check(p.graph.execute_plan_at_index(p.handle,bindings,workspace,p.selected));
    else check(p.graph.execute(p.handle,bindings,workspace));
    return 0;
  }catch(const std::exception& e){error=e.what();return -1;}
}
API void sdpa_destroy(void* p){delete static_cast<Plan*>(p);}
