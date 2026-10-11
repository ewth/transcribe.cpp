// KV cache growth through a host staging buffer.
//
// kv_grow_staged frees the old cache before allocating the new one, so the
// filled rows make a round trip through host memory and land at the new
// per-layer stride. A wrong stride or byte count leaves the decode attending
// to shifted or zeroed keys, which degrades a long transcript without
// failing, so the layout is checked directly against kv_grow.

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using transcribe::causal_lm::KvCache;

constexpr int k_heads = 2;
constexpr int k_dim   = 4;
constexpr int k_layer = 3;
constexpr int k_row   = k_heads * k_dim;

float pattern(int t, int l, int pos, int e) {
    return static_cast<float>(1 + t * 100000 + l * 10000 + pos * 100 + e);
}

// Fill every position of every layer, then mark the first `n` as written.
void fill(KvCache & cache, int n) {
    const int          n_ctx = cache.n_ctx;
    std::vector<float> host(static_cast<size_t>(k_layer) * n_ctx * k_row);
    for (int t = 0; t < 2; ++t) {
        for (int l = 0; l < k_layer; ++l) {
            for (int pos = 0; pos < n_ctx; ++pos) {
                for (int e = 0; e < k_row; ++e) {
                    host[(static_cast<size_t>(l) * n_ctx + pos) * k_row + e] = pattern(t, l, pos, e);
                }
            }
        }
        ggml_backend_tensor_set(t == 0 ? cache.self_k : cache.self_v, host.data(), 0, host.size() * sizeof(float));
    }
    cache.n    = n;
    cache.head = n;
}

std::vector<float> read(const KvCache & cache, int t) {
    std::vector<float> host(static_cast<size_t>(k_layer) * cache.n_ctx * k_row);
    ggml_backend_tensor_get(t == 0 ? cache.self_k : cache.self_v, host.data(), 0, host.size() * sizeof(float));
    return host;
}

void test_rows_survive_at_the_new_stride(ggml_backend_t backend) {
    KvCache cache;
    CHECK(transcribe::causal_lm::kv_init(cache, backend, 8, k_heads, k_dim, k_layer, GGML_TYPE_F32));
    fill(cache, 5);
    CHECK(transcribe::causal_lm::kv_grow_staged(cache, backend, 16, k_heads, k_dim, k_layer));
    CHECK(cache.n_ctx == 16);
    CHECK(cache.n == 5);
    CHECK(cache.head == 5);
    for (int t = 0; t < 2; ++t) {
        const std::vector<float> got = read(cache, t);
        for (int l = 0; l < k_layer; ++l) {
            for (int pos = 0; pos < 16; ++pos) {
                for (int e = 0; e < k_row; ++e) {
                    const float v = got[(static_cast<size_t>(l) * 16 + pos) * k_row + e];
                    // Written rows carry over; rows past the fill (written or
                    // not in the old cache) start cleared, as kv_init leaves them.
                    CHECK(v == (pos < 5 ? pattern(t, l, pos, e) : 0.0f));
                }
            }
        }
    }
    cache.free();
}

void test_matches_kv_grow(ggml_backend_t backend) {
    KvCache staged;
    KvCache direct;
    CHECK(transcribe::causal_lm::kv_init(staged, backend, 8, k_heads, k_dim, k_layer, GGML_TYPE_F32));
    CHECK(transcribe::causal_lm::kv_init(direct, backend, 8, k_heads, k_dim, k_layer, GGML_TYPE_F32));
    fill(staged, 8);
    fill(direct, 8);
    CHECK(transcribe::causal_lm::kv_grow_staged(staged, backend, 20, k_heads, k_dim, k_layer));
    CHECK(transcribe::causal_lm::kv_grow(direct, backend, 20, k_heads, k_dim, k_layer));
    for (int t = 0; t < 2; ++t) {
        CHECK(read(staged, t) == read(direct, t));
    }
    CHECK(staged.n == direct.n && staged.head == direct.head && staged.n_ctx == direct.n_ctx);
    staged.free();
    direct.free();
}

void test_refuses_to_shrink(ggml_backend_t backend) {
    KvCache cache;
    CHECK(transcribe::causal_lm::kv_init(cache, backend, 8, k_heads, k_dim, k_layer, GGML_TYPE_F16));
    cache.n = cache.head = 3;
    CHECK(!transcribe::causal_lm::kv_grow_staged(cache, backend, 8, k_heads, k_dim, k_layer));
    CHECK(!transcribe::causal_lm::kv_grow_staged(cache, backend, 4, k_heads, k_dim, k_layer));
    // Refusal leaves the cache as it was.
    CHECK(cache.ctx != nullptr && cache.n_ctx == 8 && cache.n == 3);
    cache.free();
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init\n");
        return 1;
    }
    test_rows_survive_at_the_new_stride(backend);
    test_matches_kv_grow(backend);
    test_refuses_to_shrink(backend);
    ggml_backend_free(backend);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("kv_grow_staged_unit: ok\n");
    return 0;
}
