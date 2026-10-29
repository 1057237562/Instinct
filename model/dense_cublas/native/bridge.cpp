// Small C ABI over public CUDA headers. No copied enum/opaque-struct layouts in Python.
#include <cublasLt.h>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string error;
void check(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(call) + ": cuBLAS status " + std::to_string(status));
}
#define LT(call) check((call), #call)
#define GUARD_BEGIN try { error.clear();
#define GUARD_END return 0; } catch (const std::exception& e) { error = e.what(); return -1; }

cudaDataType_t dtype(int code) {
    switch (code) {
        case 0: return CUDA_R_32F;
        case 1: return CUDA_R_16BF;
        case 2: return CUDA_R_16F;
        case 3: return CUDA_R_8F_E4M3;
        case 4: return CUDA_R_8F_E5M2;
        default: throw std::invalid_argument("unsupported dtype code");
    }
}

struct Plan {
    cublasLtHandle_t handle = nullptr; // owned by caller
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    bool fp8 = false;
    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};
} // namespace

API const char* instinct_lt_error() { return error.c_str(); }
API uint64_t instinct_lt_version() { return cublasLtGetVersion(); }
API int instinct_lt_abi() { return 2; }

API int instinct_lt_create(void** out) {
    GUARD_BEGIN
    if (!out) throw std::invalid_argument("null handle output");
    *out = nullptr;
    cublasLtHandle_t handle = nullptr;
    LT(cublasLtCreate(&handle));
    *out = handle;
    GUARD_END
}

API int instinct_lt_destroy(void* handle) {
    GUARD_BEGIN
    if (handle) LT(cublasLtDestroy(static_cast<cublasLtHandle_t>(handle)));
    GUARD_END
}

// Column-major op(A)[m,k] @ op(B)[k,n] -> C[m,n]. High-precision inputs
// use natural leading dimensions directly; FP8 uses explicitly packed TN.
static int make_plan(void* handle, int64_t m, int64_t n, int64_t k,
                         int a_type, int b_type, int out_type, uint64_t workspace,
                         int max_candidates, int sm_target, int transpose_a, int transpose_b, int fast_accum,
                         int batch, int64_t stride_a, int64_t stride_b, int64_t stride_c, void** out) {
    GUARD_BEGIN
    if (!out || !handle || m <= 0 || n <= 0 || k <= 0 || max_candidates < 1 || max_candidates > 64)
        throw std::invalid_argument("invalid matmul plan arguments");
    if (batch < 1 || (batch > 1 && (stride_a <= 0 || stride_b <= 0 || stride_c <= 0)))
        throw std::invalid_argument("invalid batched matmul strides");
    *out = nullptr;
    auto plan = std::make_unique<Plan>();
    plan->handle = static_cast<cublasLtHandle_t>(handle);
    plan->fp8 = a_type >= 3 || b_type >= 3;
    if ((a_type >= 3) != (b_type >= 3) || (a_type == 4 && b_type == 4))
        throw std::invalid_argument("FP8 requires E4M3/E4M3 or E4M3/E5M2 operands");
    if (out_type < 0 || out_type > 2)
        throw std::invalid_argument("output must be FP32/BF16/FP16");
    if ((transpose_a != 0 && transpose_a != 1) || (transpose_b != 0 && transpose_b != 1))
        throw std::invalid_argument("transpose flags must be zero or one");
    if (plan->fp8 && (!transpose_a || transpose_b))
        throw std::invalid_argument("FP8 requires column-major TN");
    auto compute = a_type == 0 && b_type == 0 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    LT(cublasLtMatmulDescCreate(&plan->operation, compute, CUDA_R_32F));
    cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    LT(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (plan->fp8) {
        int8_t fast = fast_accum ? 1 : 0;
        LT(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    }
    if (sm_target > 0)
        LT(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm_target, sizeof(sm_target)));
    const int64_t ar = transpose_a ? k : m, ac = transpose_a ? m : k;
    const int64_t br = transpose_b ? n : k, bc = transpose_b ? k : n;
    LT(cublasLtMatrixLayoutCreate(&plan->a, dtype(a_type), ar, ac, ar));
    LT(cublasLtMatrixLayoutCreate(&plan->b, dtype(b_type), br, bc, br));
    LT(cublasLtMatrixLayoutCreate(&plan->c, dtype(out_type), m, n, m));
    if (batch > 1) {
        auto set_batch = [&](cublasLtMatrixLayout_t layout, int64_t stride) {
            LT(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
            LT(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride,sizeof(stride)));
        };
        set_batch(plan->a,stride_a);set_batch(plan->b,stride_b);set_batch(plan->c,stride_c);
    }
    LT(cublasLtMatmulPreferenceCreate(&plan->preference));
    size_t limit = static_cast<size_t>(workspace);
    LT(cublasLtMatmulPreferenceSetAttribute(plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
    // cuMemAlloc guarantees >=256-byte alignment; every packed operand starts
    // at its allocation base. Do not apply this plan to offset subviews.
    uint32_t alignment = 256;
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        LT(cublasLtMatmulPreferenceSetAttribute(plan->preference, attr, &alignment, sizeof(alignment)));
    std::vector<cublasLtMatmulHeuristicResult_t> results(max_candidates);
    int count = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation, plan->a, plan->b, plan->c, plan->c,
                                     plan->preference, max_candidates, results.data(), &count));
    for (int i = 0; i < count; ++i) {
        if (results[i].state == CUBLAS_STATUS_SUCCESS && results[i].workspaceSize <= limit)
            plan->candidates.push_back(results[i]);
    }
    if (plan->candidates.empty()) throw std::runtime_error("no cuBLASLt algorithm fits precision/layout/workspace constraints");
    *out = plan.release();
    GUARD_END
}

API int instinct_lt_plan_ex(void* handle, int64_t m, int64_t n, int64_t k,
                         int a_type, int b_type, int out_type, uint64_t workspace,
                         int max_candidates, int sm_target, int transpose_a, int transpose_b, int fast_accum, void** out) {
    return make_plan(handle,m,n,k,a_type,b_type,out_type,workspace,max_candidates,sm_target,
                     transpose_a,transpose_b,fast_accum,1,0,0,0,out);
}

API int instinct_lt_plan_batched(void* handle, int64_t m, int64_t n, int64_t k,
                         int data_type, uint64_t workspace, int max_candidates,
                         int transpose_a, int transpose_b, int batch,
                         int64_t stride_a, int64_t stride_b, int64_t stride_c, void** out) {
    return make_plan(handle,m,n,k,data_type,data_type,data_type,workspace,max_candidates,0,
                     transpose_a,transpose_b,0,batch,stride_a,stride_b,stride_c,out);
}

// Identical-size BF16 tiles with device pointer arrays. Used for a triangular
// Gram decomposition; off-diagonal tiles are mirrored separately by the caller.
API int instinct_lt_plan_pointer_gram(void* handle, int tile, int reduction, int leading,
                                     int batch, uint64_t workspace, int candidates, void** out) {
    GUARD_BEGIN
    if(!handle || !out || tile<1 || reduction<1 || leading<tile || batch<1 || candidates<1 || candidates>64)
        throw std::invalid_argument("invalid pointer Gram geometry");
    auto p=std::make_unique<Plan>();p->handle=static_cast<cublasLtHandle_t>(handle);
    LT(cublasLtMatmulDescCreate(&p->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t t=CUBLAS_OP_T,n=CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t)));
    LT(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSB,&n,sizeof(n)));
    LT(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16BF,reduction,tile,reduction));
    LT(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16BF,reduction,tile,reduction));
    LT(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,tile,tile,leading));
    uint32_t mode=CUBLASLT_BATCH_MODE_POINTER_ARRAY;
    for(auto layout:{p->a,p->b,p->c}) {
        LT(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
        LT(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,&mode,sizeof(mode)));
    }
    LT(cublasLtMatmulPreferenceCreate(&p->preference));size_t limit=static_cast<size_t>(workspace);
    LT(cublasLtMatmulPreferenceSetAttribute(p->preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
    uint32_t alignment=std::gcd(uint32_t(256),uint32_t(tile)*2u);
    for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                   CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        LT(cublasLtMatmulPreferenceSetAttribute(p->preference,attr,&alignment,sizeof(alignment)));
    std::vector<cublasLtMatmulHeuristicResult_t> results(candidates);int count=0;
    LT(cublasLtMatmulAlgoGetHeuristic(p->handle,p->operation,p->a,p->b,p->c,p->c,p->preference,candidates,results.data(),&count));
    for(int i=0;i<count;++i)if(results[i].state==CUBLAS_STATUS_SUCCESS && results[i].workspaceSize<=limit)p->candidates.push_back(results[i]);
    if(p->candidates.empty())throw std::runtime_error("no pointer-array Gram algorithm on this device");
    *out=p.release();
    GUARD_END
}

// Retain ABI 2's precise-accumulation behavior for existing callers.
API int instinct_lt_plan(void* handle, int64_t m, int64_t n, int64_t k,
                         int a_type, int b_type, int out_type, uint64_t workspace,
                         int max_candidates, int sm_target, int transpose_a, int transpose_b, void** out) {
    return instinct_lt_plan_ex(handle,m,n,k,a_type,b_type,out_type,workspace,
                               max_candidates,sm_target,transpose_a,transpose_b,0,out);
}

API int instinct_lt_plan_destroy(void* opaque) {
    GUARD_BEGIN
    delete static_cast<Plan*>(opaque);
    GUARD_END
}

API int instinct_lt_candidate_count(void* opaque) {
    return opaque ? static_cast<int>(static_cast<Plan*>(opaque)->candidates.size()) : 0;
}

API int instinct_lt_candidate_info(void* opaque, int index, uint64_t* workspace, int* algo_id, float* waves) {
    GUARD_BEGIN
    if (!opaque || !workspace || !algo_id || !waves) throw std::invalid_argument("null candidate argument");
    auto& result = static_cast<Plan*>(opaque)->candidates.at(index);
    *workspace = result.workspaceSize;
    *waves = result.wavesCount;
    size_t written = 0;
    LT(cublasLtMatmulAlgoConfigGetAttribute(&result.algo, CUBLASLT_ALGO_CONFIG_ID, algo_id, sizeof(*algo_id), &written));
    GUARD_END
}

API int instinct_lt_run(void* opaque, int index, const void* a, const void* b, void* out,
                        const void* scale_a, const void* scale_b,
                        void* workspace, uint64_t workspace_bytes, void* stream) {
    GUARD_BEGIN
    if (!opaque || !a || !b || !out) throw std::invalid_argument("null matmul argument");
    auto& plan = *static_cast<Plan*>(opaque);
    auto& result = plan.candidates.at(index);
    if (result.workspaceSize > workspace_bytes || (result.workspaceSize && !workspace))
        throw std::invalid_argument("insufficient workspace");
    if (plan.fp8) {
        if (!scale_a || !scale_b) throw std::invalid_argument("FP8 requires device dequantization scales");
        LT(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
        LT(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));
    }
    const float alpha = 1.0f, beta = 0.0f;
    LT(cublasLtMatmul(plan.handle, plan.operation, &alpha, a, plan.a, b, plan.b,
                      &beta, out, plan.c, out, plan.c, &result.algo,
                      workspace, static_cast<size_t>(workspace_bytes), static_cast<cudaStream_t>(stream)));
    GUARD_END
}

API int instinct_lt_run_batched(void* opaque, int index, const void* a, const void* b,
                               const void* c, void* out, float alpha, float beta,
                               void* workspace, uint64_t workspace_bytes, void* stream) {
    GUARD_BEGIN
    if (!opaque || !a || !b || !c || !out) throw std::invalid_argument("null batched GEMM argument");
    auto& plan=*static_cast<Plan*>(opaque);
    auto& result=plan.candidates.at(index);
    if (plan.fp8) throw std::invalid_argument("optimizer batched GEMM uses high precision operands");
    if (result.workspaceSize>workspace_bytes || (result.workspaceSize && !workspace))
        throw std::invalid_argument("insufficient batched GEMM workspace");
    LT(cublasLtMatmul(plan.handle,plan.operation,&alpha,a,plan.a,b,plan.b,&beta,c,plan.c,
                     out,plan.c,&result.algo,workspace,static_cast<size_t>(workspace_bytes),
                     static_cast<cudaStream_t>(stream)));
    GUARD_END
}
