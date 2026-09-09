#include "ggml-rknpu2.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <arm_neon.h>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// RKNPU2 backend for the RK3588 NPU. Task 3: FP16 weight ownership and budget
// admission. Task 4: synchronous 2D FP16 matmul (MUL_MAT) on top of it.

// default weight budget in bytes (2 GiB), inside one 4 GiB iommu domain
#define GGML_RKNPU2_BUDGET_DEFAULT 2147483648ULL

// SDK splits B into T segments when K exceeds this; stay single-block for now
#define GGML_RKNPU2_K_MAX 8192

// RK3588 FP16 matmul alignment is in elements, not bytes.
#define GGML_RKNPU2_K_ALIGN 32
#define GGML_RKNPU2_N_ALIGN 16

// packed FP16 weight in the SDK B native layout, admitted and owned per tensor
struct ggml_backend_rknpu2_weight {
    rknn_matmul_ctx  ctx;  // owner matmul context, used to alloc/sync/free mem
    rknn_tensor_mem * mem; // packed weight buffer (rknn_create_mem2)
    size_t           size; // SDK-returned size charged against the budget
    std::vector<ggml_backend_rknpu2_weight> parts;
};

static uint32_t ggml_backend_rknpu2_part_count(uint32_t n) {
    return n >= 3 * GGML_RKNPU2_N_ALIGN ? 3 : 1;
}

static uint32_t ggml_backend_rknpu2_part_size(uint32_t n, uint32_t part) {
    const uint32_t count = ggml_backend_rknpu2_part_count(n);
    const uint32_t blocks = n / GGML_RKNPU2_N_ALIGN;
    return (blocks / count + (part < blocks % count)) * GGML_RKNPU2_N_ALIGN;
}

// execution-context cache key: token rows, K, partition columns, and core
struct ggml_backend_rknpu2_shape_key {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t core = 0;

    bool operator<(const ggml_backend_rknpu2_shape_key & other) const {
        if (M != other.M) {
            return M < other.M;
        }
        if (K != other.K) {
            return K < other.K;
        }
        if (N != other.N) {
            return N < other.N;
        }
        return core < other.core;
    }
};

struct ggml_backend_rknpu2_activation {
    rknn_matmul_ctx ctx;
    rknn_tensor_mem * mem;
    size_t size;
    size_t refs;
};

// per-shape execution context: an rknn matmul context plus its bound A/C scratch
struct ggml_backend_rknpu2_exec {
    rknn_matmul_ctx     ctx;
    rknn_matmul_io_attr io;       // A/B/C sizes and dims; B attr reused to bind weights
    rknn_tensor_mem *   memA;     // shared FP16 A scratch
    rknn_tensor_mem *   memC;     // F32 C scratch, owned by this context
    size_t              scratch; // C bytes charged to used_scratch
    rknn_tensor_mem *   boundB;   // weight mem currently bound as B (nullptr = unbound)
    ggml_backend_rknpu2_activation * activation;
};

class ggml_backend_rknpu2_worker {
    std::mutex mutex;
    std::condition_variable ready;
    rknn_matmul_ctx context = 0;
    bool pending = false;
    bool stopping = false;
    int result = RKNN_SUCC;
    std::thread thread;

public:
    ggml_backend_rknpu2_worker() : thread([this] {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            ready.wait(lock, [this] { return pending || stopping; });
            if (stopping) {
                return;
            }
            const auto current = context;
            lock.unlock();
            const int status = rknn_matmul_run(current);
            lock.lock();
            result = status;
            pending = false;
            ready.notify_all();
        }
    }) {}

    ~ggml_backend_rknpu2_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ready.notify_all();
        }
        thread.join();
    }

    void submit(rknn_matmul_ctx next) {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(!pending);
        context = next;
        pending = true;
        ready.notify_all();
    }

    int wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [this] { return !pending; });
        return result;
    }
};

// backend-wide state shared by every buffer and stream of this device
struct ggml_backend_rknpu2_context {
    size_t budget;          // GGML_RKNPU2_BUDGET bytes, guarded-admission ceiling
    size_t used_persistent; // packed weight buffers + their owner matmul contexts
    size_t used_scratch;    // cached execution contexts + their A/C scratch

    // packed weights keyed by tensor pointer, never by name
    std::map<const struct ggml_tensor *, ggml_backend_rknpu2_weight> weights;

    // per-shape execution contexts cached until device teardown
    std::map<ggml_backend_rknpu2_shape_key, ggml_backend_rknpu2_exec *> execs;
    std::map<std::pair<uint32_t, uint32_t>, ggml_backend_rknpu2_activation *> activations;

    std::mutex mutex; // serializes every backend entry point over the shared context
    int        refs;  // live ggml_backend instances
    std::unique_ptr<ggml_backend_rknpu2_worker> workers[2];
};

// host buffer context: the shared backend state plus the host allocation
struct ggml_backend_rknpu2_buffer_context {
    struct ggml_backend_rknpu2_context * ctx;
    void * data;
};

static size_t ggml_backend_rknpu2_budget_from_env(void) {
    const char * env = getenv("GGML_RKNPU2_BUDGET");
    if (env == nullptr) {
        return GGML_RKNPU2_BUDGET_DEFAULT;
    }

    char * end  = nullptr;
    errno       = 0;
    unsigned long long value = strtoull(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0') {
        GGML_LOG_ERROR("%s: invalid GGML_RKNPU2_BUDGET %s, using default\n", __func__, env);
        return GGML_RKNPU2_BUDGET_DEFAULT;
    }

    return (size_t) value;
}

static int64_t ggml_backend_rknpu2_min_m(void) {
    static const int64_t minimum = [] {
        const char * env = getenv("GGML_RKNPU2_MIN_M");
        if (env == nullptr) {
            return int64_t(2);
        }
        char * end = nullptr;
        errno = 0;
        const long long value = strtoll(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || value < 1 || value > INT32_MAX) {
            GGML_LOG_ERROR("%s: invalid GGML_RKNPU2_MIN_M, using 2\n", __func__);
            return int64_t(2);
        }
        return int64_t(value);
    }();
    return minimum;
}

static struct ggml_backend_rknpu2_context * ggml_backend_rknpu2_context_get(void) {
    static struct ggml_backend_rknpu2_context ctx = {
        /* .budget          = */ ggml_backend_rknpu2_budget_from_env(),
        /* .used_persistent = */ 0,
        /* .used_scratch    = */ 0,
        /* .weights         = */ {},
        /* .execs           = */ {},
        /* .activations     = */ {},
        /* .mutex           = */ {},
        /* .refs            = */ 0,
        /* .workers         = */ {},
    };

    return &ctx;
}

// source types we can convert to FP16: F32 directly, anything else via to_float
static bool ggml_backend_rknpu2_weight_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }
    return ggml_get_type_traits(type)->to_float != nullptr;
}

// only packed at admission; a graph never recomputes a NONE leaf, so its bytes are stable
static bool ggml_backend_rknpu2_is_constant_weight(const struct ggml_tensor * w) {
    const struct ggml_tensor * leaf = w;
    while (leaf != nullptr && leaf->view_src != nullptr) {
        leaf = leaf->view_src;
    }
    return leaf != nullptr && leaf->op == GGML_OP_NONE;
}

// pure shape/type eligibility for a 2D FP16 MUL_MAT: no admission, no side effects
static bool ggml_backend_rknpu2_mul_mat_eligible(const struct ggml_tensor * op) {
    const struct ggml_tensor * w = op->src[0]; // weights   [K, N]
    const struct ggml_tensor * a = op->src[1]; // activations [K, M]

    if (w->ne[2] != 1 || w->ne[3] != 1 || a->ne[2] != 1 || a->ne[3] != 1) {
        return false; // 2D only, reject batch/broadcast
    }
    if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32) {
        return false; // F32 result and F32 activations only
    }
    if (!ggml_backend_rknpu2_weight_type_supported(w->type)) {
        return false; // no FP16 conversion path for the weight
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(a)) {
        return false;
    }
    if (ggml_get_op_params_i32(op, 1) != GGML_HINT_NONE) {
        return false; // Hadamard and other hint-dispatched variants unsupported
    }

    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    const int64_t M = a->ne[1];
    if (a->ne[0] != K || K < 1 || N < 1 || M < 1 || N > INT32_MAX || M > INT32_MAX) {
        return false;
    }
    if (K > GGML_RKNPU2_K_MAX) {
        return false; // SDK T-segment split above 8192
    }
    if (K % GGML_RKNPU2_K_ALIGN != 0 || N % GGML_RKNPU2_N_ALIGN != 0) {
        return false; // native B tile alignment
    }

    return true;
}

// guarded subtraction; zero budget and over-budget requests are denied
static bool ggml_backend_rknpu2_budget_admit(struct ggml_backend_rknpu2_context * ctx, size_t n) {
    if (n == 0 || n > ctx->budget) {
        return false;
    }
    const size_t used_total = ctx->used_persistent + ctx->used_scratch;
    return n <= ctx->budget - used_total;
}

static void ggml_backend_rknpu2_weight_release(struct ggml_backend_rknpu2_context * ctx, ggml_backend_rknpu2_weight * w) {
    for (auto & part : w->parts) {
        ggml_backend_rknpu2_weight_release(ctx, &part);
    }
    w->parts.clear();
    if (w->mem != nullptr) {
        // drop any surviving exec B binding to this mem before it is freed
        for (const auto & pair : ctx->execs) {
            if (pair.second->boundB == w->mem) {
                pair.second->boundB = nullptr;
            }
        }
        rknn_destroy_mem(w->ctx, w->mem);
    }
    if (w->ctx != 0) {
        rknn_matmul_destroy(w->ctx);
    }

    GGML_ASSERT(ctx->used_persistent >= w->size);
    ctx->used_persistent -= w->size;

    w->mem  = nullptr;
    w->ctx  = 0;
    w->size = 0;
}

static void ggml_backend_rknpu2_invalidate_weight(struct ggml_backend_rknpu2_context * ctx, const struct ggml_tensor * w) {
    auto it = ctx->weights.find(w);
    if (it != ctx->weights.end()) {
        ggml_backend_rknpu2_weight_release(ctx, &it->second);
        ctx->weights.erase(it);
    }
}

static bool ggml_backend_rknpu2_tensor_in_buffer(const struct ggml_tensor * w, ggml_backend_buffer_t buffer) {
    return w->buffer == buffer || (w->view_src != nullptr && w->view_src->buffer == buffer);
}

static void ggml_backend_rknpu2_invalidate_buffer(struct ggml_backend_rknpu2_context * ctx, ggml_backend_buffer_t buffer) {
    for (auto it = ctx->weights.begin(); it != ctx->weights.end();) {
        if (ggml_backend_rknpu2_tensor_in_buffer(it->first, buffer)) {
            ggml_backend_rknpu2_weight_release(ctx, &it->second);
            it = ctx->weights.erase(it);
        } else {
            ++it;
        }
    }
}

// convert one weight tensor to FP16 in the SDK B native layout (N/subN, K/subK, subN, subK)
static bool ggml_backend_rknpu2_pack_weight(
        const struct ggml_tensor * w,
        void * packed,
        size_t packed_size,
        uint32_t subN,
        uint32_t subK) {
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];

    const uint32_t n_blocks = (uint32_t) ((N + subN - 1) / subN);
    const uint32_t k_blocks = (uint32_t) ((K + subK - 1) / subK);

    const size_t expected = (size_t) n_blocks * k_blocks * subN * subK * sizeof(ggml_fp16_t);
    GGML_ASSERT(expected <= packed_size);

    // zero all padding slots, then scatter the real values over them
    memset(packed, 0, packed_size);

    ggml_fp16_t * dst = (ggml_fp16_t *) packed;
    ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;

    std::vector<float> row(K);

    for (int64_t n = 0; n < N; ++n) {
        const char * src = (const char *) w->data + n * w->nb[1];

        if (w->type == GGML_TYPE_F32) {
            const float * f = (const float *) src;
            for (int64_t k = 0; k < K; ++k) {
                row[k] = f[k];
            }
        } else {
            to_float(src, row.data(), K);
        }

        const uint32_t n_block = (uint32_t) (n / subN);
        const uint32_t nn      = (uint32_t) (n % subN);

        for (int64_t k = 0; k < K; ++k) {
            const float v = row[k];
            if (!std::isfinite(v) || v > 65504.0f || v < -65504.0f) {
                GGML_LOG_ERROR("%s: non-finite or non-FP16-representable value in weight of type %s\n", __func__, ggml_type_name(w->type));
                return false;
            }

            const uint32_t k_block = (uint32_t) (k / subK);
            const uint32_t kk      = (uint32_t) (k % subK);
            const size_t idx = (((size_t) n_block * k_blocks + k_block) * subN + nn) * subK + kk;
            dst[idx] = ggml_fp32_to_fp16(v);
        }
    }

    return true;
}

// admit a weight: SDK-returned packed size, guarded budget, alloc + write + sync
static enum ggml_status ggml_backend_rknpu2_admit_part(struct ggml_backend_rknpu2_context * ctx, struct ggml_tensor * w, ggml_backend_rknpu2_weight & result) {
    if (!ggml_backend_rknpu2_weight_type_supported(w->type)) {
        return GGML_STATUS_ALLOC_FAILED;
    }

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M     = 1; // placeholder; B layout/size are independent of M
    info.K     = (int32_t) w->ne[0];
    info.N     = (int32_t) w->ne[1];
    info.type  = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;
    info.iommu_domain_id = 0;

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(io_attr));

    rknn_matmul_ctx mctx = 0;
    int ret = rknn_matmul_create(&mctx, &info, &io_attr);
    if (ret != RKNN_SUCC || mctx == 0) {
        GGML_LOG_ERROR("%s: rknn_matmul_create failed ret=%d\n", __func__, ret);
        return GGML_STATUS_ALLOC_FAILED;
    }

    // charge the SDK-returned packed size, which includes native-layout padding
    const size_t packed_size = io_attr.B.size;
    if (!ggml_backend_rknpu2_budget_admit(ctx, packed_size)) {
        GGML_LOG_WARN("%s: weight budget denied for tensor of type %s (size %zu), CPU fallback\n", __func__, ggml_type_name(w->type), packed_size);
        rknn_matmul_destroy(mctx);
        return GGML_STATUS_ALLOC_FAILED;
    }

    rknn_tensor_mem * mem = rknn_create_mem2(mctx, packed_size, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
    if (mem == nullptr) {
        GGML_LOG_ERROR("%s: rknn_create_mem2 failed for %zu bytes\n", __func__, packed_size);
        rknn_matmul_destroy(mctx);
        return GGML_STATUS_ALLOC_FAILED;
    }
    ctx->used_persistent += packed_size;

    if (!ggml_backend_rknpu2_pack_weight(w, mem->virt_addr, packed_size, io_attr.B.dims[2], io_attr.B.dims[3])) {
        rknn_destroy_mem(mctx, mem);
        rknn_matmul_destroy(mctx);
        ctx->used_persistent -= packed_size;
        return GGML_STATUS_FAILED;
    }

    ret = rknn_mem_sync(mctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN_SUCC) {
        GGML_LOG_ERROR("%s: rknn_mem_sync TO_DEVICE failed ret=%d\n", __func__, ret);
        rknn_destroy_mem(mctx, mem);
        rknn_matmul_destroy(mctx);
        ctx->used_persistent -= packed_size;
        return GGML_STATUS_ALLOC_FAILED;
    }

    result = ggml_backend_rknpu2_weight { mctx, mem, packed_size, {} };
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_rknpu2_admit_weight(struct ggml_backend_rknpu2_context * ctx, struct ggml_tensor * w) {
    if (ctx->weights.find(w) != ctx->weights.end()) {
        return GGML_STATUS_SUCCESS;
    }
    ggml_backend_rknpu2_weight result = {};
    const uint32_t count = ggml_backend_rknpu2_part_count(w->ne[1]);
    result.parts.resize(count);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        ggml_tensor part = *w;
        part.ne[1] = ggml_backend_rknpu2_part_size(w->ne[1], i);
        part.data = (char *) w->data + offset * w->nb[1];
        const ggml_status status = ggml_backend_rknpu2_admit_part(ctx, &part, result.parts[i]);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_rknpu2_weight_release(ctx, &result);
            return status;
        }
        offset += part.ne[1];
    }
    ctx->weights.emplace(w, std::move(result));
    return GGML_STATUS_SUCCESS;
}

// for Task 4: the admitted packed weight, or nullptr when not admitted
static const ggml_backend_rknpu2_weight * ggml_backend_rknpu2_weight_lookup(
        struct ggml_backend_rknpu2_context * ctx, const struct ggml_tensor * w) {
    auto it = ctx->weights.find(w);
    return it == ctx->weights.end() ? nullptr : &it->second;
}

// release one execution context and credit its scratch back to the budget
static void ggml_backend_rknpu2_exec_release(
        struct ggml_backend_rknpu2_context * ctx, ggml_backend_rknpu2_shape_key key) {
    auto it = ctx->execs.find(key);
    if (it == ctx->execs.end()) {
        return;
    }

    ggml_backend_rknpu2_exec * e = it->second;
    ctx->execs.erase(it);

    auto * activation = e->activation;
    if (e->memC != nullptr) {
        rknn_destroy_mem(e->ctx, e->memC);
    }
    if (e->ctx != activation->ctx) {
        rknn_matmul_destroy(e->ctx);
    }

    GGML_ASSERT(ctx->used_scratch >= e->scratch);
    ctx->used_scratch -= e->scratch;

    // Keep the allocation's SDK owner alive after its execution entry is removed.
    if (--activation->refs == 0) {
        ctx->activations.erase({ key.M, key.K });
        rknn_destroy_mem(activation->ctx, activation->mem);
        rknn_matmul_destroy(activation->ctx);
        GGML_ASSERT(ctx->used_scratch >= activation->size);
        ctx->used_scratch -= activation->size;
        delete activation;
    }

    delete e;
}

// release all admitted weights and exec contexts; caller holds ctx->mutex
static void ggml_backend_rknpu2_context_reset_locked(ggml_backend_rknpu2_context * ctx) {
    for (auto & worker : ctx->workers) {
        worker.reset();
    }
    for (auto it = ctx->weights.begin(); it != ctx->weights.end();) {
        ggml_backend_rknpu2_weight_release(ctx, &it->second);
        it = ctx->weights.erase(it);
    }
    while (!ctx->execs.empty()) {
        ggml_backend_rknpu2_exec_release(ctx, ctx->execs.begin()->first);
    }
    GGML_ASSERT(ctx->activations.empty());
    ctx->used_persistent = 0;
    ctx->used_scratch = 0;
}

// get (or create) the execution context for one shape; scratch is admitted at
// placement and released at device teardown - the memory budget is the only bound.
// Returns nullptr when scratch admission or any SDK call fails (the op must run
// on CPU).
static ggml_backend_rknpu2_exec * ggml_backend_rknpu2_exec_get(
        struct ggml_backend_rknpu2_context * ctx, const ggml_backend_rknpu2_shape_key & key) {
    if (key.core > 0 && !ctx->workers[key.core - 1]) {
        try {
            ctx->workers[key.core - 1].reset(new ggml_backend_rknpu2_worker);
        } catch (const std::exception & error) {
            GGML_LOG_ERROR("%s: worker creation failed: %s\n", __func__, error.what());
            return nullptr;
        }
    }
    auto it = ctx->execs.find(key);
    if (it != ctx->execs.end()) {
        return it->second;
    }

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M     = (int32_t) key.M;
    info.K     = (int32_t) key.K;
    info.N     = (int32_t) key.N;
    info.type  = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout  = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;
    info.iommu_domain_id = 0;

    ggml_backend_rknpu2_exec * e = new ggml_backend_rknpu2_exec;
    memset(e, 0, sizeof(*e));

    int ret = rknn_matmul_create(&e->ctx, &info, &e->io);
    if (ret != RKNN_SUCC || e->ctx == 0) {
        GGML_LOG_ERROR("%s: rknn_matmul_create failed ret=%d M=%u K=%u N=%u\n", __func__, ret, key.M, key.K, key.N);
        delete e;
        return nullptr;
    }

    ret = rknn_matmul_set_core_mask(e->ctx, (rknn_core_mask) (1u << key.core));
    if (ret != RKNN_SUCC) {
        GGML_LOG_ERROR("%s: rknn_matmul_set_core_mask failed ret=%d\n", __func__, ret);
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    const auto activation_key = std::make_pair(key.M, key.K);
    const auto input = ctx->activations.find(activation_key);
    const bool new_activation = input == ctx->activations.end();
    if (!new_activation && input->second->size != e->io.A.size) {
        GGML_LOG_ERROR("%s: inconsistent shared A size M=%u K=%u\n", __func__, key.M, key.K);
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }
    const size_t scratch = (new_activation ? (size_t) e->io.A.size : 0) + (size_t) e->io.C.size;
    if (!ggml_backend_rknpu2_budget_admit(ctx, scratch)) {
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    e->memA = new_activation ? rknn_create_mem2(e->ctx, e->io.A.size, RKNN_FLAG_MEMORY_FLAGS_DEFAULT) : input->second->mem;
    if (e->memA == nullptr) {
        GGML_LOG_ERROR("%s: rknn_create_mem2 A failed for %u bytes\n", __func__, e->io.A.size);
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    e->memC = rknn_create_mem2(e->ctx, e->io.C.size, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
    if (e->memC == nullptr) {
        GGML_LOG_ERROR("%s: rknn_create_mem2 C failed for %u bytes\n", __func__, e->io.C.size);
        if (new_activation) {
            rknn_destroy_mem(e->ctx, e->memA);
        }
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    ret = rknn_matmul_set_io_mem(e->ctx, e->memA, &e->io.A);
    if (ret != RKNN_SUCC) {
        GGML_LOG_ERROR("%s: rknn_matmul_set_io_mem A failed ret=%d\n", __func__, ret);
        rknn_destroy_mem(e->ctx, e->memC);
        if (new_activation) {
            rknn_destroy_mem(e->ctx, e->memA);
        }
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    ret = rknn_matmul_set_io_mem(e->ctx, e->memC, &e->io.C);
    if (ret != RKNN_SUCC) {
        GGML_LOG_ERROR("%s: rknn_matmul_set_io_mem C failed ret=%d\n", __func__, ret);
        rknn_destroy_mem(e->ctx, e->memC);
        if (new_activation) {
            rknn_destroy_mem(e->ctx, e->memA);
        }
        rknn_matmul_destroy(e->ctx);
        delete e;
        return nullptr;
    }

    if (new_activation) {
        e->activation = new ggml_backend_rknpu2_activation { e->ctx, e->memA, e->io.A.size, 0 };
        ctx->activations.emplace(activation_key, e->activation);
    } else {
        e->activation = input->second;
    }
    ++e->activation->refs;
    e->scratch = e->io.C.size;
    ctx->used_scratch += scratch;
    ctx->execs.emplace(key, e);

    return e;
}

// host buffer: CPU-allocated slice in standard ggml layout, owned by this backend

static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    uintptr_t data = (uintptr_t) bufctx->data;
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }
    return (void *) data;
}

static void ggml_backend_rknpu2_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    if (bufctx == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
        ggml_backend_rknpu2_invalidate_buffer(bufctx->ctx, buffer);
    }
    ggml_aligned_free(bufctx->data, buffer->size);
    delete bufctx;
}

static void ggml_backend_rknpu2_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
    memset((char *) tensor->data + offset, value, size);
    ggml_backend_rknpu2_invalidate_weight(bufctx->ctx, tensor);
}

static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
    memcpy((char *) tensor->data + offset, data, size);
    ggml_backend_rknpu2_invalidate_weight(bufctx->ctx, tensor);
}

static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_rknpu2_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
        std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
        memcpy(dst->data, src->data, ggml_nbytes(src));
        ggml_backend_rknpu2_invalidate_weight(bufctx->ctx, dst);
        return true;
    }
    return false;
}

static void ggml_backend_rknpu2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
    memset(bufctx->data, value, buffer->size);
    ggml_backend_rknpu2_invalidate_buffer(bufctx->ctx, buffer);
}

static void ggml_backend_rknpu2_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * bufctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bufctx->ctx->mutex);
    ggml_backend_rknpu2_invalidate_buffer(bufctx->ctx, buffer);
}

static const struct ggml_backend_buffer_i ggml_backend_rknpu2_buffer_i = {
    /* .free_buffer     = */ ggml_backend_rknpu2_buffer_free,
    /* .get_base        = */ ggml_backend_rknpu2_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_rknpu2_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rknpu2_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rknpu2_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_rknpu2_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rknpu2_buffer_clear,
    /* .reset           = */ ggml_backend_rknpu2_buffer_reset,
};

// host buffer type (default): CPU-allocated host memory in standard ggml layout

static const char * ggml_backend_rknpu2_buft_name(ggml_backend_buffer_type_t buft) {
    return "RKNPU2_HOST";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_rknpu2_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return nullptr;
    }

    ggml_backend_rknpu2_buffer_context * bufctx = new ggml_backend_rknpu2_buffer_context;
    bufctx->ctx  = ggml_backend_rknpu2_context_get();
    bufctx->data = data;

    return ggml_backend_buffer_init(buft, ggml_backend_rknpu2_buffer_i, bufctx, size);
}

static size_t ggml_backend_rknpu2_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_rknpu2_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_buft_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_rknpu2_buft_name,
            /* .alloc_buffer     = */ ggml_backend_rknpu2_buft_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_rknpu2_buft_get_alignment,
            /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_rknpu2_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rknpu2_reg(), 0),
        /* .context = */ nullptr,
    };

    return &buft;
}

// backend (stream) interface - MUL_MAT runs on the NPU

static const char * ggml_backend_rknpu2_get_name(ggml_backend_t backend) {
    return "RKNPU2";

    GGML_UNUSED(backend);
}

static void ggml_backend_rknpu2_convert_activation(const float * src, ggml_fp16_t * dst, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t values = vld1q_f32(src + i);
        const uint16x4_t halves = vreinterpret_u16_f16(vcvt_f16_f32(values));
        // Match GGML's signed canonical NaN instead of retaining the payload.
        const uint16x4_t nan_mask = vmovn_u32(vmvnq_u32(vceqq_f32(values, values)));
        const uint16x4_t sign = vand_u16(vshrn_n_u32(vreinterpretq_u32_f32(values), 16), vdup_n_u16(0x8000));
        const uint16x4_t nan = vorr_u16(sign, vdup_n_u16(0x7e00));
        vst1_u16(dst + i, vbsl_u16(nan_mask, nan, halves));
    }
    ggml_fp32_to_fp16_row(src + i, dst + i, n - i);
}

static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    struct ggml_backend_rknpu2_context * ctx = (struct ggml_backend_rknpu2_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (--ctx->refs == 0) {
            ggml_backend_rknpu2_context_reset_locked(ctx);
        }
    }
    delete backend;
}

static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_rknpu2_context * ctx = (struct ggml_backend_rknpu2_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * dst = cgraph->nodes[i];
        if (dst->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * w = dst->src[0];       // weights   [K, N]
        const struct ggml_tensor * a = dst->src[1]; // activations [K, M]

        const int64_t K = w->ne[0];
        const int64_t N = w->ne[1];
        const int64_t M = a->ne[1];

        if (M < 1 || K < 1 || N < 1 || K > GGML_RKNPU2_K_MAX) {
            return GGML_STATUS_FAILED;
        }

        const ggml_backend_rknpu2_weight * wgt = ggml_backend_rknpu2_weight_lookup(ctx, w);
        if (wgt == nullptr) {
            if (!ggml_backend_rknpu2_is_constant_weight(w) ||
                w->buffer == nullptr || w->buffer->buft != ggml_backend_rknpu2_buft_type() ||
                !ggml_backend_rknpu2_mul_mat_eligible(dst)) {
                GGML_LOG_ERROR("%s: MUL_MAT weight not admissible (non-constant or ineligible) K=%lld N=%lld\n", __func__, (long long) K, (long long) N);
                return GGML_STATUS_FAILED;
            }
            if (ggml_backend_rknpu2_admit_weight(ctx, w) != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("%s: weight admission failed for MUL_MAT K=%lld N=%lld\n", __func__, (long long) K, (long long) N);
                return GGML_STATUS_FAILED;
            }
            wgt = ggml_backend_rknpu2_weight_lookup(ctx, w);
            GGML_ASSERT(wgt != nullptr);
        }

        ggml_backend_rknpu2_exec * executions[3] = {};
        const uint32_t count = ggml_backend_rknpu2_part_count(N);
        for (uint32_t part = 0; part < count; ++part) {
            const ggml_backend_rknpu2_shape_key key = { (uint32_t) M, (uint32_t) K, ggml_backend_rknpu2_part_size(N, part), part };
            ggml_backend_rknpu2_exec * e = ggml_backend_rknpu2_exec_get(ctx, key);
            if (e == nullptr) {
                // scratch admission denied or an SDK call failed: fail rather than fabricate
                return GGML_STATUS_FAILED;
            }
            executions[part] = e;
            const auto & weight_part = wgt->parts[part];

            // (re)bind the weight's B memory when it changed (re-admission or shape reuse)
            if (e->boundB != weight_part.mem) {
                int ret = rknn_matmul_set_io_mem(e->ctx, weight_part.mem, &e->io.B);
                if (ret != RKNN_SUCC) {
                    GGML_LOG_ERROR("%s: rknn_matmul_set_io_mem B failed ret=%d\n", __func__, ret);
                    ggml_backend_rknpu2_exec_release(ctx, key);
                    return GGML_STATUS_FAILED;
                }
                e->boundB = weight_part.mem;
            }
        }

        // All cores read one normal-layout [M, K] activation, refreshed per op.
        auto * activation = executions[0]->activation;
        ggml_fp16_t * adst = (ggml_fp16_t *) activation->mem->virt_addr;
        const float * asrc = (const float *) a->data;
        const size_t arow = a->nb[1] / sizeof(float);
        for (int64_t m = 0; m < M; ++m) {
            ggml_backend_rknpu2_convert_activation(asrc + m * arow, adst + m * K, K);
        }

        const int sync_ret = rknn_mem_sync(activation->ctx, activation->mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (sync_ret != RKNN_SUCC) {
            GGML_LOG_ERROR("%s: rknn_mem_sync A failed ret=%d\n", __func__, sync_ret);
            return GGML_STATUS_FAILED;
        }

        for (uint32_t part = 1; part < count; ++part) {
            ctx->workers[part - 1]->submit(executions[part]->ctx);
        }
        int status[3] = { rknn_matmul_run(executions[0]->ctx), RKNN_SUCC, RKNN_SUCC };
        for (uint32_t part = 1; part < count; ++part) {
            status[part] = ctx->workers[part - 1]->wait();
        }
        bool failed = false;
        for (uint32_t part = 0; part < count; ++part) {
            if (status[part] != RKNN_SUCC) {
                GGML_LOG_ERROR("%s: rknn_matmul_run failed core=%u ret=%d\n", __func__, part, status[part]);
                failed = true;
            }
        }
        if (failed) {
            return GGML_STATUS_FAILED;
        }
        for (uint32_t part = 0; part < count; ++part) {
            auto * e = executions[part];
            int ret = rknn_mem_sync(e->ctx, e->memC, RKNN_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN_SUCC) {
                GGML_LOG_ERROR("%s: rknn_mem_sync C failed ret=%d\n", __func__, ret);
                return GGML_STATUS_FAILED;
            }
        }

        uint32_t offset = 0;
        for (uint32_t part = 0; part < count; ++part) {
            auto * e = executions[part];
            const uint32_t columns = ggml_backend_rknpu2_part_size(N, part);

            // C is F32 normal layout [M, N]; ggml dst [N, M] shares that memory order
            const float * csrc = (const float *) e->memC->virt_addr;
            float * dptr = (float *) dst->data;
            const size_t drow = dst->nb[1] / sizeof(float);
            for (int64_t m = 0; m < M; ++m) {
                memcpy(dptr + m * drow + offset, csrc + m * columns, (size_t) columns * sizeof(float));
            }
            offset += columns;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_rknpu2_backend_i = {
    /* .get_name                = */ ggml_backend_rknpu2_get_name,
    /* .free                    = */ ggml_backend_rknpu2_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_rknpu2_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    static ggml_guid guid = { 0x52, 0x4b, 0x4e, 0x50, 0x55, 0x32, 0x52, 0x4b, 0x33, 0x35, 0x38, 0x38, 0x4e, 0x50, 0x55, 0x01 };
    return &guid;
}

static ggml_backend_t ggml_backend_rknpu2_init(void) {
    struct ggml_backend_rknpu2_context * ctx = ggml_backend_rknpu2_context_get();
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ++ctx->refs;
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rknpu2_guid(),
        /* .iface   = */ ggml_backend_rknpu2_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rknpu2_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

// device interface

static const char * ggml_backend_rknpu2_device_get_name(ggml_backend_dev_t dev) {
    return "RKNPU2";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_rknpu2_device_get_description(ggml_backend_dev_t dev) {
    return "Rockchip RK3588 NPU (RKNPU2 runtime)";

    GGML_UNUSED(dev);
}

static void ggml_backend_rknpu2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // the SDK has no portable free-memory query; report none rather than the software budget
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_rknpu2_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_rknpu2_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknpu2_device_get_name(dev);
    props->description = ggml_backend_rknpu2_device_get_description(dev);
    props->type        = ggml_backend_rknpu2_device_get_type(dev);
    ggml_backend_rknpu2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rknpu2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknpu2_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rknpu2_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_rknpu2_buft_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_rknpu2_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_rknpu2_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    struct ggml_backend_rknpu2_context * ctx = ggml_backend_rknpu2_context_get();
    std::lock_guard<std::mutex> lock(ctx->mutex);

    if (op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_NONE) {
        return true; // compute-free leaf tensor probed by test-backend-ops
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    struct ggml_tensor * w = op->src[0];       // weights   [K, N]
    const struct ggml_tensor * a = op->src[1]; // activations [K, M]

    if (!ggml_backend_rknpu2_mul_mat_eligible(op)) {
        return false;
    }

    // model-loader capability probe: zero-size dummy buffer, no data yet
    if (w->data == nullptr) {
        return true;
    }

    if (a->ne[1] < ggml_backend_rknpu2_min_m()) {
        return false;
    }

    // MANDATE B: never admit a weight outside our own host buffer type, whose
    // invalidation hooks cannot see writes (mmap or plain CPU buffers)
    if (w->buffer == nullptr || w->buffer->buft != ggml_backend_rknpu2_buft_type()) {
        return false;
    }

    // only packed, stable leaf weights are admitted
    if (!ggml_backend_rknpu2_is_constant_weight(w)) {
        return false;
    }

    // MANDATE C: skip re-admission when the weight is already present
    const bool new_weight = ggml_backend_rknpu2_weight_lookup(ctx, w) == nullptr;
    if (new_weight && !ggml_backend_rknpu2_budget_admit(ctx, (size_t) w->ne[0] * w->ne[1] * sizeof(ggml_fp16_t))) {
        return false;
    }

    // reserve the execution context + A/C scratch now; scratch denial or an SDK
    // shape rejection then falls back to CPU at placement, not at graph_compute
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    std::vector<ggml_backend_rknpu2_shape_key> created;
    for (uint32_t part = 0; part < ggml_backend_rknpu2_part_count(N); ++part) {
        const ggml_backend_rknpu2_shape_key key = { (uint32_t) a->ne[1], (uint32_t) K, ggml_backend_rknpu2_part_size(N, part), part };
        const bool existing = ctx->execs.find(key) != ctx->execs.end();
        if (ggml_backend_rknpu2_exec_get(ctx, key) == nullptr) {
            // roll back this incomplete admission, never an earlier reservation
            for (const auto & fresh : created) {
                ggml_backend_rknpu2_exec_release(ctx, fresh);
            }
            return false;
        }
        if (!existing) {
            created.push_back(key);
        }
    }
    if (new_weight && ggml_backend_rknpu2_admit_weight(ctx, w) != GGML_STATUS_SUCCESS) {
        for (const auto & fresh : created) {
            ggml_backend_rknpu2_exec_release(ctx, fresh);
        }
        return false;
    }
    return true;
}

static bool ggml_backend_rknpu2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_rknpu2_device_i = {
    /* .get_name             = */ ggml_backend_rknpu2_device_get_name,
    /* .get_description      = */ ggml_backend_rknpu2_device_get_description,
    /* .get_memory           = */ ggml_backend_rknpu2_device_get_memory,
    /* .get_type             = */ ggml_backend_rknpu2_device_get_type,
    /* .get_props            = */ ggml_backend_rknpu2_device_get_props,
    /* .init_backend         = */ ggml_backend_rknpu2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknpu2_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_rknpu2_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rknpu2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknpu2_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// backend registration

static const char * ggml_backend_rknpu2_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNPU2";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_rknpu2_reg_get_device_count(ggml_backend_reg_t reg) {
    // the SDK is linked in, so the device is present whenever this backend is built
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rknpu2_device = {
        /* .iface   = */ ggml_backend_rknpu2_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_backend_rknpu2_context_get(),
    };

    return &ggml_backend_rknpu2_device;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rknpu2_reg_i = {
    /* .get_name         = */ ggml_backend_rknpu2_reg_get_name,
    /* .get_device_count = */ ggml_backend_rknpu2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rknpu2_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static struct ggml_backend_reg ggml_backend_rknpu2_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rknpu2_reg_i,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_rknpu2_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
