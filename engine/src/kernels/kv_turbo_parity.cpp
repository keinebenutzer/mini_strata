// src/kernels/kv_turbo_parity.cpp - Parity test for the 2-bit turbo2 and 3-bit turbo3 KV storage
// (include/strata/kernels/kv_turbo.hpp and kv_turbo3.hpp). Validates, all BITWISE:
// 1. The group quantizer and packing (32 values per block, 10 bytes in turbo2 / 14 in turbo3 with its 3-bit
//    little-endian code bitstream) vs a host reference that runs the identical float expressions (IEEE division,
//    rintf round-half-to-even, f16_bits.hpp's converter) - random gaussian rows plus crafted tie / all-zero /
//    subnormal-scale blocks.
// 2. kv_append_turbo2_step / _steps / the prompt batch (with staging pool and host-copy mirror) against the host
//    reference through a permuted page table, page_size 8, several pages - and the same for the turbo3 forms.
// 3. kv_gather_turbo2_step / kv_gather_turbo3_step: the FP16 scratch is bit-identical to a host dequant of the
//    stored blocks, with the gather ids shuffled across pages.
// 4. The planes=1 fold (v == k, the K8V4-style call shape) stores each group exactly once.
// 5. k3v8 (`--kv k3v8`, kv_k3v8.hpp): the asymmetric mode has no kernels of its own - it folds each half onto
//    its pool and calls the single-format kernels (the kKvHybrid trick). run_k3v8_phases proves exactly that new
//    wiring bitwise: folded turbo3 appends + folded int8 appends write the pools AND the host copy (through the
//    kv_k3v8_*_half views) like the host quantizers, and the folded gathers read both halves back bitwise.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_turbo.hpp"
#include "strata/kernels/kv_turbo3.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_k3v8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// Host reference for one 32-value block: the same expressions as kv_turbo.cu's turbo_group, in the same order.
// d = fp16(amax / 0.75); code = clamp(rint(x / (d/2) + 1.5), 0, 3); element j in bits 2*(j%4) of qs[j/4].
void quantize_block_turbo2_host(const float* x, k::block_turbo2& blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    blk.d = k::f16_from_f32(amax / 0.75f);
    const float hd = k::f32_from_f16(blk.d) * 0.5f;
    for (int j = 0; j < 8; ++j) {
        uint8_t byte = 0;
        for (int m = 0; m < 4; ++m) {
            const float v = x[4 * j + m];
            const float t = hd != 0.0f ? v / hd : 0.0f;
            int q = (int) std::rint(t + 1.5f);
            q = q < 0 ? 0 : (q > 3 ? 3 : q);
            byte |= (uint8_t) (q << (2 * m));
        }
        blk.qs[j] = byte;
    }
}

// Host reference for the gather: dequant one stored block into 32 fp16 values (the kv_gather kernel's expressions).
void dequant_block_turbo2_host(const k::block_turbo2& blk, uint16_t* out) {
    const float hd = k::f32_from_f16(blk.d) * 0.5f;
    for (int t = 0; t < 32; ++t) {
        const int code = (blk.qs[t >> 2] >> ((t & 3) << 1)) & 3;
        out[t] = k::f16_from_f32(((float) code - 1.5f) * hd);
    }
}

// Host reference for one 32-value turbo3 block: the same expressions as kv_turbo3.cu's turbo3_group, in the same
// order. d = fp16(amax / 0.875); code = clamp(rint(x / (d*0.25) + 3.5), 0, 7); the 32 codes packed as a 96-bit
// little-endian bitstream: value i in bits 3i..3i+2 of qs[] (crossing byte boundaries at shifts 6 and 7).
void quantize_block_turbo3_host(const float* x, k::block_turbo3& blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    blk.d = k::f16_from_f32(amax / 0.875f);
    const float hd = k::f32_from_f16(blk.d) * 0.25f;
    uint8_t bits[12] = {0};
    for (int i = 0; i < 32; ++i) {
        const float t = hd != 0.0f ? x[i] / hd : 0.0f;
        int q = (int) std::rint(t + 3.5f);
        q = q < 0 ? 0 : (q > 7 ? 7 : q);
        const int bit = 3 * i, by = bit >> 3, sh = bit & 7;
        bits[by] |= (uint8_t) (q << sh);
        if (sh > 5) bits[by + 1] |= (uint8_t) (q >> (8 - sh));
    }
    for (int j = 0; j < 12; ++j) blk.qs[j] = bits[j];
}

// Host reference for the turbo3 gather: dequant one stored block into 32 fp16 values.
void dequant_block_turbo3_host(const k::block_turbo3& blk, uint16_t* out) {
    const float hd = k::f32_from_f16(blk.d) * 0.25f;
    for (int t = 0; t < 32; ++t) {
        const int by = (3 * t) >> 3, sh = (3 * t) & 7;
        const int code = sh <= 5 ? ((blk.qs[by] >> sh) & 7)
                                 : ((((int) blk.qs[by] >> sh) | ((int) blk.qs[by + 1] << (8 - sh))) & 7);
        out[t] = k::f16_from_f32(((float) code - 3.5f) * hd);
    }
}

// Host reference for one 64-value int8 group: the same expressions as kv_q8.cu's append kernel - sbits =
// fp16(amax / 127), q = clamp(__float2int_rn(x / fp16_to_f32(sbits)), -127, 127) (rintf is the same RN mode).
void quant_group_q8_host(const float* x, int8_t* codes, uint16_t* scale_bits) {
    float amax = 0.0f;
    for (int i = 0; i < k::KV_Q8_GROUP; ++i) amax = fmaxf(amax, std::fabs(x[i]));
    const uint16_t sbits = k::f16_from_f32(amax / 127.0f);
    const float sf = k::f32_from_f16(sbits);
    for (int i = 0; i < k::KV_Q8_GROUP; ++i) {
        int q = 0;
        if (sf > 0.0f) {
            q = (int) rintf(x[i] / sf);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        codes[i] = (int8_t) q;
    }
    *scale_bits = sbits;
}

long long row_of(const std::vector<int32_t>& table, int pos, int h, int H, int P) {
    return ((long long) table[pos / P] * H + h) * P + (pos % P);
}

}  // namespace

// The turbo3 phases, mirroring main()'s turbo2 ones one for one with the 3-bit pools: per-token append, the
// multi-step / prompt-batch forms with host copy + staging pool + the planes=1 fold, the gather, and the
// reconstruction bound. All bitwise vs the host references above.
void run_turbo3_phases(std::mt19937& rng, std::normal_distribution<float>& nd) {
    static_assert(sizeof(k::block_turbo3) == 14, "block_turbo3 must be 14 bytes");
    if (k::kv_turbo3_bytes_per_head(256) != 112 || k::kv_turbo3_bytes_per_cell(k::qsa_real_shapes()) != 448) {
        std::fprintf(stderr, "FAIL: turbo3 bytes_per_head/cell helpers wrong\n");
        ++g_fail;
        return;
    }

    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 8;
    const int H = (int) s.n_head_kv;       // 2
    const int D = (int) s.head_dim;        // 256
    const int P = (int) s.page_size;       // 8
    const int pages = 12;
    const int cells = pages * P;
    const int blocks_per_head = D / k::KV_TURBO3_GROUP;                    // 8
    const int bytes_per_head = blocks_per_head * (int) sizeof(k::block_turbo3);   // 112

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 7 + 5) % pages;   // a different derangement than turbo2's
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy t3 table");

    const size_t pool_bytes = (size_t) pages * H * P * bytes_per_head;
    uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v = dalloc<uint8_t>(pool_bytes);

    // -------------------------------------------------------------
    // [T1/4] Per-token append through the paged pool, bitwise vs the host quantizer
    // -------------------------------------------------------------
    std::printf("[T1/4] kv_append_turbo3_step: pool blocks bitwise vs host quantizer...\n");
    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<char> filled(cells, 0);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    const int n_fill = cells - 9;   // leave some cells never written
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;
        // Crafted groups ride on the first head's first blocks: exact ties (amax 1.75 -> d 2.0, hd 0.5, so
        // x/hd + 3.5 lands on .5 boundaries), an all-zero block, and a subnormal-scale block.
        if (n % 7 == 0) {
            for (int i = 0; i < 32; ++i) kv[i] = (i % 2 ? -1.0f : 1.75f) * (1.0f - 0.02f * i);   // ties near 0.5 steps
            for (int i = 32; i < 64; ++i) kv[i] = 0.0f;                                          // zero block
            for (int i = 64; i < 96; ++i) kv[i] = (i & 1 ? -1e-7f : 1.3e-7f);                   // subnormal d
        }
        if (n % 5 == 0) {
            for (int i = 0; i < 32; ++i) vv[i] = (i % 4 == 0 ? -1.75f : (i % 4 == 2 ? 1.0f : 0.25f));   // exact ties
        }
        host_k[pos] = kv;
        host_v[pos] = vv;
        filled[pos] = 1;

        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy t3 step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy t3 k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy t3 v");
        k::kv_append_turbo3_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_turbo3_step");
    }

    std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
    ck(cudaMemcpy(h_k.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "memcpy t3 d2h k");
    ck(cudaMemcpy(h_v.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "memcpy t3 d2h v");

    long bad_blocks = 0, checked_blocks = 0;
    for (int pos = 0; pos < cells; ++pos) {
        if (!filled[pos]) continue;
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, pos, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const float* raw = (is_v ? host_v[pos] : host_k[pos]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo3 ref;
                    quantize_block_turbo3_host(raw + b * 32, ref);
                    const k::block_turbo3* gpu = reinterpret_cast<const k::block_turbo3*>(pool_ptr) + b;
                    ++checked_blocks;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo3)) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at pos=%d is_v=%d h=%d b=%d\n", pos, is_v, h, b);
                            std::printf("  host d=0x%04x (%g)  gpu d=0x%04x (%g)\n", ref.d, k::f32_from_f16(ref.d),
                                        gpu->d, k::f32_from_f16(gpu->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 12; ++i) std::printf("%02x ", ref.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 12; ++i) std::printf("%02x ", gpu->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
    }
    if (bad_blocks) {
        std::fprintf(stderr, "FAIL: %ld of %ld turbo3 blocks differ from the host quantizer\n", bad_blocks,
                     checked_blocks);
        ++g_fail;
    } else {
        std::printf("  -> all %ld turbo3 pool blocks bitwise equal to the host quantizer (OK)\n", checked_blocks);
    }

    // -------------------------------------------------------------
    // [T2/4] Multi-step append, the prompt batch (stage + host mirror), and the planes=1 fold
    // -------------------------------------------------------------
    std::printf("[T2/4] kv_append_turbo3_steps / prompt batch with stage + host copy / planes fold...\n");
    const int T = 40;   // spans 5 pages
    const int64_t pos0 = 0;
    std::vector<float> K((size_t) T * H * D), V((size_t) T * H * D);
    for (auto& x : K) x = nd(rng) * 1.5f;
    for (auto& x : V) x = nd(rng) * 1.5f;
    float* d_K = dalloc<float>(K.size());
    float* d_V = dalloc<float>(V.size());
    ck(cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "memcpy t3 K");
    ck(cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "memcpy t3 V");

    uint8_t *h_k_pool = nullptr, *h_v_pool = nullptr, *d_host_k = nullptr, *d_host_v = nullptr;
    ck(cudaHostAlloc(&h_k_pool, pool_bytes, cudaHostAllocMapped), "hostalloc t3 k");
    ck(cudaHostAlloc(&h_v_pool, pool_bytes, cudaHostAllocMapped), "hostalloc t3 v");
    ck(cudaHostGetDevicePointer((void**) &d_host_k, h_k_pool, 0), "hostptr t3 k");
    ck(cudaHostGetDevicePointer((void**) &d_host_v, h_v_pool, 0), "hostptr t3 v");
    ck(cudaMemset(h_k_pool, 0xFF, pool_bytes), "memset t3 host k");
    ck(cudaMemset(h_v_pool, 0xFF, pool_bytes), "memset t3 host v");
    uint8_t* d_stage_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_stage_v = dalloc<uint8_t>(pool_bytes);

    uint8_t* d_k2 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v2 = dalloc<uint8_t>(pool_bytes);
    k::KvHostPools host_pools;
    host_pools.k_turbo3 = d_host_k;
    host_pools.v_turbo3 = d_host_v;
    k::KvHostPools stage_pools;
    stage_pools.k_turbo3 = d_stage_k;
    stage_pools.v_turbo3 = d_stage_v;
    k::kv_append_turbo3(d_k2, d_v2, d_table, pos0, T, d_K, d_V, s, nullptr, &host_pools, &stage_pools);
    ck(cudaDeviceSynchronize(), "kv_append_turbo3");

    uint8_t* d_k3 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v3 = dalloc<uint8_t>(pool_bytes);
    std::vector<int32_t> steps_dev((size_t) T * k::kStepCount);
    for (int t = 0; t < T; ++t) {
        steps_dev[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) (pos0 + t);
        steps_dev[(size_t) t * k::kStepCount + k::kStepWidth] = 1;
    }
    int32_t* d_steps = dalloc<int32_t>(steps_dev.size());
    ck(cudaMemcpy(d_steps, steps_dev.data(), steps_dev.size() * 4, cudaMemcpyHostToDevice), "memcpy t3 steps");
    k::kv_append_turbo3_steps(d_k3, d_v3, d_table, d_steps, k::kStepCount, T, d_K, d_V, s, nullptr);
    ck(cudaDeviceSynchronize(), "kv_append_turbo3_steps");

    uint8_t* d_fold = dalloc<uint8_t>(pool_bytes);
    k::kv_append_turbo3_steps(d_fold, d_fold, d_table, d_steps, k::kStepCount, T, d_K, d_K, s, nullptr);
    ck(cudaDeviceSynchronize(), "kv_append_turbo3 fold");

    std::vector<uint8_t> h_k2(pool_bytes), h_v2(pool_bytes), h_k3(pool_bytes), h_v3(pool_bytes), h_fold(pool_bytes);
    ck(cudaMemcpy(h_k2.data(), d_k2, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 k2");
    ck(cudaMemcpy(h_v2.data(), d_v2, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 v2");
    ck(cudaMemcpy(h_k3.data(), d_k3, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 k3");
    ck(cudaMemcpy(h_v3.data(), d_v3, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 v3");
    ck(cudaMemcpy(h_fold.data(), d_fold, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 fold");

    long bad2 = 0;
    for (int t = 0; t < T; ++t) {
        const int pos = (int) (pos0 + t);
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, pos, h, H, P);
                const float* raw = (is_v ? V : K).data() + ((size_t) t * H + h) * D;
                const uint8_t* pool_arr[2] = {(is_v ? h_v2.data() : h_k2.data()), (is_v ? h_v3.data() : h_k3.data())};
                for (int which = 0; which < 2; ++which) {
                    const uint8_t* pool_ptr = pool_arr[which] + row * bytes_per_head;
                    for (int b = 0; b < blocks_per_head; ++b) {
                        k::block_turbo3 ref;
                        quantize_block_turbo3_host(raw + b * 32, ref);
                        const k::block_turbo3* gpu = reinterpret_cast<const k::block_turbo3*>(pool_ptr) + b;
                        if (std::memcmp(&ref, gpu, sizeof(k::block_turbo3)) != 0) ++bad2;
                    }
                }
                const long long hrow = (((long long) pos / P) * H + h) * P + (pos % P);
                const uint8_t* hp = (is_v ? h_v_pool : h_k_pool) + hrow * bytes_per_head;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo3 ref;
                    quantize_block_turbo3_host(raw + b * 32, ref);
                    const k::block_turbo3* gpu = reinterpret_cast<const k::block_turbo3*>(hp) + b;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo3)) != 0) ++bad2;
                }
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo3 ref;
                    quantize_block_turbo3_host(K.data() + ((size_t) t * H + h) * D + b * 32, ref);
                    const k::block_turbo3* gpu = reinterpret_cast<const k::block_turbo3*>(h_fold.data() + row * bytes_per_head) + b;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo3)) != 0) ++bad2;
                }
            }
        }
    }
    std::vector<uint8_t> h_sk(pool_bytes), h_sv(pool_bytes);
    ck(cudaMemcpy(h_sk.data(), d_stage_k, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 stage k");
    ck(cudaMemcpy(h_sv.data(), d_stage_v, pool_bytes, cudaMemcpyDeviceToHost), "d2h t3 stage v");
    for (int t = 0; t < T; ++t) {
        const int pos = (int) (pos0 + t);
        for (int h = 0; h < H; ++h) {
            const long long srow = (((long long) pos / P) * H + h) * P + (pos % P);
            for (int is_v = 0; is_v < 2; ++is_v) {
                const uint8_t* src = (is_v ? h_sv.data() : h_sk.data()) + srow * bytes_per_head;
                const uint8_t* dst = (is_v ? h_v2.data() : h_k2.data()) + row_of(table, pos, h, H, P) * bytes_per_head;
                if (std::memcmp(src, dst, bytes_per_head) != 0) ++bad2;
            }
        }
    }
    if (bad2) {
        std::fprintf(stderr, "FAIL: %ld mismatches in the turbo3 batch / steps / host-copy / staging / fold checks\n", bad2);
        ++g_fail;
    } else {
        std::printf("  -> turbo3 batch, steps, host copy, staging pool and the planes=1 fold all bitwise (OK)\n");
    }

    // -------------------------------------------------------------
    // [T3/4] Gather: FP16 scratch bitwise vs a host dequant of the stored blocks
    // -------------------------------------------------------------
    std::printf("[T3/4] kv_gather_turbo3_step: FP16 scratch bitwise vs host dequant...\n");
    const int max_ids = 96;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scr = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scr = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = positions[rng() % n_fill];   // shuffled across pages
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy t3 ids");
    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy t3 step");

    k::kv_gather_turbo3_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, d_k_scr, d_v_scr, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_turbo3_step");

    std::vector<uint16_t> h_k_scr((size_t) max_ids * H * D), h_v_scr((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_k_scr.data(), d_k_scr, h_k_scr.size() * 2, cudaMemcpyDeviceToHost), "d2h t3 k_scr");
    ck(cudaMemcpy(h_v_scr.data(), d_v_scr, h_v_scr.size() * 2, cudaMemcpyDeviceToHost), "d2h t3 v_scr");

    long bad_gather = 0;
    double worst_err = 0.0, worst_rel = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, cell, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const uint16_t* got = (is_v ? h_v_scr.data() : h_k_scr.data()) + (size_t) (i * H + h) * D;
                const float* raw = (is_v ? host_v[cell] : host_k[cell]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    const k::block_turbo3* stored = reinterpret_cast<const k::block_turbo3*>(pool_ptr) + b;
                    uint16_t ref[32];
                    dequant_block_turbo3_host(*stored, ref);
                    if (std::memcmp(ref, got + b * 32, 64) != 0) ++bad_gather;
                    for (int t = 0; t < 32; ++t) {
                        const double err = std::fabs(k::f32_from_f16(got[b * 32 + t]) - raw[t + b * 32]);
                        if (err > worst_err) worst_err = err;
                        const double scale = std::fabs(k::f32_from_f16(stored->d));
                        if (scale > 1e-6 && err / scale > worst_rel) worst_rel = err / scale;
                    }
                }
            }
        }
    }
    if (bad_gather) {
        std::fprintf(stderr, "FAIL: %ld gathered turbo3 values differ from the host dequant\n", bad_gather);
        ++g_fail;
    } else {
        std::printf("  -> turbo3 gather bitwise vs host dequant (OK); max |x'-x| = %.4f, max relative to the block "
                    "scale = %.4f (centroids 0.25*d apart)\n", worst_err, worst_rel);
    }

    // -------------------------------------------------------------
    // [T4/4] Reconstruction sanity: 3-bit after the rotation stays bounded
    // -------------------------------------------------------------
    std::printf("[T4/4] turbo3 reconstruction bound: every dequantized value within 0.25*d of its input...\n");
    double bound = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, cell, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const uint16_t* got = (is_v ? h_v_scr.data() : h_k_scr.data()) + (size_t) (i * H + h) * D;
                const float* raw = (is_v ? host_v[cell] : host_k[cell]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    const float d = k::f32_from_f16(reinterpret_cast<const k::block_turbo3*>(pool_ptr)[b].d);
                    for (int t = 0; t < 32; ++t) {
                        const double err = std::fabs(k::f32_from_f16(got[b * 32 + t]) - raw[t + b * 32]);
                        // centroids are +-0.125*d .. +-0.875*d in 0.25*d steps; the worst rounding error is half
                        // the step (the outer centroid additionally absorbs the fp16 rounding of d, ~0.05% of d)
                        if (err > bound) bound = err;
                        if (d > 1e-6f && err > 0.25 * (double) d + 1e-3) {
                            std::fprintf(stderr, "FAIL: turbo3 value %g dequantized to %g with scale %g\n",
                                         (double) raw[t + b * 32], (double) k::f32_from_f16(got[b * 32 + t]), (double) d);
                            ++g_fail;
                        }
                    }
                }
            }
        }
    }
    std::printf("  -> turbo3 max reconstruction error %.4f (bound 0.25*d holds, OK)\n", bound);

    cudaFree(d_table); cudaFree(d_k); cudaFree(d_v); cudaFree(d_k2); cudaFree(d_v2); cudaFree(d_k3); cudaFree(d_v3);
    cudaFree(d_fold); cudaFree(d_stage_k); cudaFree(d_stage_v); cudaFree(d_K); cudaFree(d_V); cudaFree(d_steps);
    cudaFree(d_ids); cudaFree(d_k_scr); cudaFree(d_v_scr); cudaFree(d_kcur); cudaFree(d_vcur); cudaFree(d_step);
    ck(cudaFreeHost(h_k_pool), "free t3 host k");
    ck(cudaFreeHost(h_v_pool), "free t3 host v");
}

// k3v8 (`--kv k3v8`, kv_k3v8.hpp): the asymmetric mode rides the single-format kernels with each half folded
// onto its pool (the kKvHybrid trick, layer.cpp). This section proves the new wiring bitwise: the folded appends
// write the VRAM pools and the pinned host copy (through the kv_k3v8_*_half views) exactly like the host
// quantizers, and the folded gathers read both halves back into the FP16 scratch bit-identical to a host dequant.
void run_k3v8_phases(std::mt19937& rng, std::normal_distribution<float>& nd) {
    if (k::kv_k3v8_bytes_per_head(256) != 376 || k::kv_k3v8_bytes_per_cell(k::qsa_real_shapes()) != 752) {
        std::fprintf(stderr, "FAIL: k3v8 bytes_per_head/cell helpers wrong\n");
        ++g_fail;
        return;
    }
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 8;
    const int H = (int) s.n_head_kv;       // 2
    const int D = (int) s.head_dim;        // 256
    const int P = (int) s.page_size;       // 8
    const int pages = 12;
    const int cells = pages * P;
    const int t3_bytes = (int) k::kv_turbo3_bytes_per_head(D);   // 112
    const int groups = D / k::KV_Q8_GROUP;                       // 4

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 11 + 7) % pages;   // a third derangement
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy k38 table");

    const size_t rows = (size_t) pages * H * P;
    uint8_t* d_k3 = dalloc<uint8_t>(rows * t3_bytes);
    int8_t* d_vq = dalloc<int8_t>(rows * D);
    uint16_t* d_vs = dalloc<uint16_t>(rows * groups);
    uint8_t* h_k3 = nullptr; int8_t* h_vq = nullptr; uint16_t* h_vs = nullptr;
    ck(cudaHostAlloc(&h_k3, rows * t3_bytes, cudaHostAllocDefault), "host k38 k");
    ck(cudaHostAlloc(&h_vq, rows * D, cudaHostAllocDefault), "host k38 v");
    ck(cudaHostAlloc(&h_vs, rows * groups * 2, cudaHostAllocDefault), "host k38 s");
    k::KvHostPools full;
    full.k_turbo3 = h_k3; full.v_q = h_vq; full.v_scale = h_vs;
    const k::KvHostPools hk = k::kv_k3v8_k_half(full), hv = k::kv_k3v8_v_half(full);

    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<char> filled(cells, 0);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    // -------------------------------------------------------------
    // [K1/3] Folded per-token appends: pools and host copy bitwise vs the host quantizers
    // -------------------------------------------------------------
    std::printf("[K1/3] k3v8 folded appends: pools + host copy bitwise vs host quantizers...\n");
    const int n_fill = cells - 9;   // leave some cells never written
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;
        if (n % 7 == 0) {   // crafted int8 groups: exact ties, an all-zero group, a subnormal scale
            for (int i = 0; i < 64; ++i) vv[i] = (i % 2 ? -1.0f : 1.75f) * (1.0f - 0.01f * i);
            for (int i = 64; i < 128; ++i) vv[i] = 0.0f;
            for (int i = 128; i < 192; ++i) vv[i] = (i & 1 ? -1e-7f : 1.3e-7f);
        }
        if (n % 5 == 0) {   // crafted turbo3 ties on the K side
            for (int i = 0; i < 32; ++i) kv[i] = (i % 4 == 0 ? -1.75f : (i % 4 == 2 ? 1.0f : 0.25f));
        }
        host_k[pos] = kv; host_v[pos] = vv; filled[pos] = 1;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy k38 step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k38 k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k38 v");
        k::kv_append_turbo3_step(d_k3, d_k3, d_table, d_step, d_kcur, d_kcur, s, nullptr, &hk);
        ck(cudaDeviceSynchronize(), "k3v8 append K");
        k::kv_append_q8_step(d_vq, d_vq, d_vs, d_vs, d_table, d_step, d_vcur, d_vcur, s, nullptr, &hv);
        ck(cudaDeviceSynchronize(), "k3v8 append V");
    }

    std::vector<uint8_t> g_k3(rows * t3_bytes);
    std::vector<int8_t> g_vq(rows * D);   // int8, not uint8: the codes are signed
    std::vector<uint16_t> g_vs(rows * groups);
    ck(cudaMemcpy(g_k3.data(), d_k3, g_k3.size(), cudaMemcpyDeviceToHost), "d2h k38 k3");
    ck(cudaMemcpy(g_vq.data(), d_vq, g_vq.size(), cudaMemcpyDeviceToHost), "d2h k38 vq");
    ck(cudaMemcpy(g_vs.data(), d_vs, g_vs.size() * 2, cudaMemcpyDeviceToHost), "d2h k38 vs");

    long bad = 0, checked = 0;
    for (int pos = 0; pos < cells; ++pos) {
        if (!filled[pos]) continue;
        for (int h = 0; h < H; ++h) {
            const long long row = row_of(table, pos, h, H, P);
            const long long hrow = ((long long) (pos / P) * H + h) * P + (pos % P);   // host copy: identity layout
            const float* raw_k = host_k[pos].data() + h * D;
            const float* raw_v = host_v[pos].data() + h * D;
            for (int b = 0; b < D / 32; ++b) {
                k::block_turbo3 ref;
                quantize_block_turbo3_host(raw_k + b * 32, ref);
                ++checked;
                if (std::memcmp(&ref, reinterpret_cast<const k::block_turbo3*>(g_k3.data() + row * t3_bytes) + b,
                                sizeof(k::block_turbo3)) != 0) ++bad;
                if (std::memcmp(&ref, reinterpret_cast<const k::block_turbo3*>(h_k3 + hrow * t3_bytes) + b,
                                sizeof(k::block_turbo3)) != 0) ++bad;
            }
            for (int g = 0; g < groups; ++g) {
                int8_t ref_codes[k::KV_Q8_GROUP]; uint16_t ref_scale = 0;
                quant_group_q8_host(raw_v + g * k::KV_Q8_GROUP, ref_codes, &ref_scale);
                ++checked;
                if (std::memcmp(ref_codes, g_vq.data() + row * D + g * k::KV_Q8_GROUP, k::KV_Q8_GROUP) != 0) ++bad;
                if (g_vs[row * groups + g] != ref_scale) ++bad;
                if (std::memcmp(ref_codes, h_vq + hrow * D + g * k::KV_Q8_GROUP, k::KV_Q8_GROUP) != 0) ++bad;
                if (h_vs[hrow * groups + g] != ref_scale) ++bad;
            }
        }
    }
    if (bad) {
        std::fprintf(stderr, "FAIL: %ld of %ld k3v8 writes differ from the host quantizers\n", bad, checked);
        ++g_fail;
    } else {
        std::printf("  -> all %ld k3v8 pool+host writes bitwise equal to the host quantizers (OK)\n", checked);
    }

    // -------------------------------------------------------------
    // [K2/3] Folded gathers: FP16 scratch bitwise vs a host dequant, ids shuffled across pages
    // -------------------------------------------------------------
    std::printf("[K2/3] k3v8 folded gathers: FP16 scratch bitwise vs host dequant...\n");
    const int max_ids = 96;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scr = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scr = dalloc<uint16_t>((size_t) max_ids * H * D);
    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = positions[rng() % n_fill];
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy k38 ids");
    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy k38 gstep");
    k::kv_gather_turbo3_step(d_k3, d_k3, d_table, d_ids, d_step, max_ids, s, d_k_scr, d_k_scr, nullptr);
    ck(cudaDeviceSynchronize(), "k3v8 gather K");
    k::kv_gather_q8_step(d_vq, d_vq, d_vs, d_vs, d_table, d_ids, d_step, max_ids, s, d_v_scr, d_v_scr, nullptr);
    ck(cudaDeviceSynchronize(), "k3v8 gather V");

    std::vector<uint16_t> g_ks((size_t) max_ids * H * D), g_vscr((size_t) max_ids * H * D);
    ck(cudaMemcpy(g_ks.data(), d_k_scr, g_ks.size() * 2, cudaMemcpyDeviceToHost), "d2h k38 kscr");
    ck(cudaMemcpy(g_vscr.data(), d_v_scr, g_vscr.size() * 2, cudaMemcpyDeviceToHost), "d2h k38 vscr");

    long bad_g = 0; double worst_q8 = 0.0, worst_q8_rel = 0.0;
    int dbg = 0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int h = 0; h < H; ++h) {
            const long long row = row_of(table, cell, h, H, P);
            const uint16_t* got_k = g_ks.data() + (size_t) (i * H + h) * D;
            const uint16_t* got_v = g_vscr.data() + (size_t) (i * H + h) * D;
            const float* raw_v = host_v[cell].data() + h * D;
            for (int b = 0; b < D / 32; ++b) {
                uint16_t ref[32];
                dequant_block_turbo3_host(*(reinterpret_cast<const k::block_turbo3*>(g_k3.data() + row * t3_bytes) + b),
                                          ref);
                if (std::memcmp(ref, got_k + b * 32, 64) != 0) {
                    ++bad_g;
                    if (dbg++ < 3) {
                        std::printf("DBG K i=%d cell=%d h=%d b=%d ref0=%.4f ref1=%.4f got0=%.4f got1=%.4f\n",
                                    i, cell, h, b, k::f32_from_f16(ref[0]), k::f32_from_f16(ref[1]),
                                    k::f32_from_f16(got_k[b * 32]), k::f32_from_f16(got_k[b * 32 + 1]));
                    }
                }
            }
            for (int g = 0; g < groups; ++g) {
                const uint16_t sf_bits = g_vs[row * groups + g];
                const float sf = k::f32_from_f16(sf_bits);
                uint16_t ref[64];
                for (int t = 0; t < k::KV_Q8_GROUP; ++t) {
                    ref[t] = k::f16_from_f32((float) g_vq[row * D + g * k::KV_Q8_GROUP + t] * sf);
                    const double err = std::fabs(k::f32_from_f16(ref[t]) - raw_v[g * k::KV_Q8_GROUP + t]);
                    if (err > worst_q8) worst_q8 = err;
                    double amax = 0.0;
                    for (int u = 0; u < k::KV_Q8_GROUP; ++u)
                        amax = std::max(amax, (double) std::fabs(raw_v[g * k::KV_Q8_GROUP + u]));
                    if (amax > 1e-6 && err / amax > worst_q8_rel) worst_q8_rel = err / amax;
                }
                if (std::memcmp(ref, got_v + g * k::KV_Q8_GROUP, 2 * k::KV_Q8_GROUP) != 0) {
                    ++bad_g;
                    if (dbg++ < 6) {
                        std::printf("DBG V i=%d cell=%d h=%d g=%d sf=%.6f ref0=%.4f ref1=%.4f got0=%.4f got1=%.4f raw0=%.4f\n",
                                    i, cell, h, g, sf, k::f32_from_f16(ref[0]), k::f32_from_f16(ref[1]),
                                    k::f32_from_f16(got_v[g * k::KV_Q8_GROUP]),
                                    k::f32_from_f16(got_v[g * k::KV_Q8_GROUP + 1]),
                                    raw_v[g * k::KV_Q8_GROUP]);
                    }
                }
            }
        }
    }
    if (bad_g) {
        std::fprintf(stderr, "FAIL: %ld k3v8 gathered values differ from the host dequant\n", bad_g);
        ++g_fail;
    } else {
        std::printf("  -> k3v8 gathers bitwise vs host dequant (OK)\n");
    }

    // -------------------------------------------------------------
    // [K3/3] int8 V reconstruction bound: rounding keeps every value within half a step of its group scale
    // -------------------------------------------------------------
    std::printf("[K3/3] k3v8 int8 V reconstruction bound...\n");
    if (worst_q8_rel > 0.008) {   // 0.5/127 ~ 0.00394 plus the fp16 scale rounding - 0.008 is the room
        std::fprintf(stderr, "FAIL: k3v8 int8 V error %.6f*amax exceeds the 0.008*amax bound\n", worst_q8_rel);
        ++g_fail;
    } else {
        std::printf("  -> max int8 V reconstruction error %.4f (abs), %.6f * group amax (bound 0.008, OK)\n",
                    worst_q8, worst_q8_rel);
    }

    cudaFree(d_table); cudaFree(d_k3); cudaFree(d_vq); cudaFree(d_vs); cudaFree(d_kcur); cudaFree(d_vcur);
    cudaFree(d_step); cudaFree(d_ids); cudaFree(d_k_scr); cudaFree(d_v_scr);
    ck(cudaFreeHost(h_k3), "free k38 h_k3");
    ck(cudaFreeHost(h_vq), "free k38 h_vq");
    ck(cudaFreeHost(h_vs), "free k38 h_vs");
}

int main() {
    std::printf("=== Running kv_turbo_parity test ===\n");
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    static_assert(sizeof(k::block_turbo2) == 10, "block_turbo2 must be 10 bytes");
    if (k::kv_turbo_bytes_per_head(256) != 80 || k::kv_turbo_bytes_per_cell(k::qsa_real_shapes()) != 320) {
        std::fprintf(stderr, "FAIL: bytes_per_head/cell helpers wrong\n");
        return 1;
    }

    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 8;                       // several pages per sequence, as the task asks
    const int H = (int) s.n_head_kv;       // 2
    const int D = (int) s.head_dim;        // 256
    const int P = (int) s.page_size;       // 8
    const int pages = 12;
    const int cells = pages * P;
    const int blocks_per_head = D / k::KV_TURBO_GROUP;   // 8
    const int bytes_per_head = blocks_per_head * (int) sizeof(k::block_turbo2);   // 80

    // Permuted page table: logical page -> physical page, a derangement so gather crosses real pages.
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    const size_t pool_bytes = (size_t) pages * H * P * bytes_per_head;
    uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v = dalloc<uint8_t>(pool_bytes);

    // -------------------------------------------------------------
    // [1/4] Per-token append through the paged pool, bitwise vs the host quantizer
    // -------------------------------------------------------------
    std::printf("[1/4] kv_append_turbo2_step: pool blocks bitwise vs host quantizer...\n");
    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<char> filled(cells, 0);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    const int n_fill = cells - 9;   // leave some cells never written
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;
        // Crafted groups ride on the first head's first blocks: exact ties (amax 1.5 -> d 2.0, hd 1.0),
        // an all-zero block, and a subnormal-scale block.
        if (n % 7 == 0) {
            for (int i = 0; i < 32; ++i) kv[i] = (i % 2 ? -1.0f : 1.5f) * (1.0f - 0.02f * i);   // ties near 0.5/2.5
            for (int i = 32; i < 64; ++i) kv[i] = 0.0f;                                          // zero block
            for (int i = 64; i < 96; ++i) kv[i] = (i & 1 ? -1e-7f : 1.3e-7f);                   // subnormal d
        }
        if (n % 5 == 0) {
            for (int i = 0; i < 32; ++i) vv[i] = (i % 4 == 0 ? -1.5f : (i % 4 == 2 ? 1.0f : 0.25f));   // exact ties
        }
        host_k[pos] = kv;
        host_v[pos] = vv;
        filled[pos] = 1;

        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");
        k::kv_append_turbo2_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_turbo2_step");
    }

    std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
    ck(cudaMemcpy(h_k.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k");
    ck(cudaMemcpy(h_v.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v");

    long bad_blocks = 0, checked_blocks = 0;
    for (int pos = 0; pos < cells; ++pos) {
        if (!filled[pos]) continue;
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, pos, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const float* raw = (is_v ? host_v[pos] : host_k[pos]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo2 ref;
                    quantize_block_turbo2_host(raw + b * 32, ref);
                    const k::block_turbo2* gpu = reinterpret_cast<const k::block_turbo2*>(pool_ptr) + b;
                    ++checked_blocks;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo2)) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at pos=%d is_v=%d h=%d b=%d\n", pos, is_v, h, b);
                            std::printf("  host d=0x%04x (%g)  gpu d=0x%04x (%g)\n", ref.d, k::f32_from_f16(ref.d),
                                        gpu->d, k::f32_from_f16(gpu->d));
                            std::printf("  host qs: ");
                            for (int i = 0; i < 8; ++i) std::printf("%02x ", ref.qs[i]);
                            std::printf("\n  gpu  qs: ");
                            for (int i = 0; i < 8; ++i) std::printf("%02x ", gpu->qs[i]);
                            std::printf("\n");
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
    }
    if (bad_blocks) {
        std::fprintf(stderr, "FAIL: %ld of %ld turbo2 blocks differ from the host quantizer\n", bad_blocks,
                     checked_blocks);
        ++g_fail;
    } else {
        std::printf("  -> all %ld pool blocks bitwise equal to the host quantizer (OK)\n", checked_blocks);
    }

    // -------------------------------------------------------------
    // [2/4] Multi-step append, the prompt batch (stage + host mirror), and the planes=1 fold
    // -------------------------------------------------------------
    std::printf("[2/4] kv_append_turbo2_steps / prompt batch with stage + host copy / planes fold...\n");
    const int T = 40;   // spans 5 pages
    const int64_t pos0 = 0;
    std::vector<float> K((size_t) T * H * D), V((size_t) T * H * D);
    for (auto& x : K) x = nd(rng) * 1.5f;
    for (auto& x : V) x = nd(rng) * 1.5f;
    float* d_K = dalloc<float>(K.size());
    float* d_V = dalloc<float>(V.size());
    ck(cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "memcpy K");
    ck(cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "memcpy V");

    // Pinned, device-mapped host copy and a staging pool, both in the identity layout.
    uint8_t *h_k_pool = nullptr, *h_v_pool = nullptr, *d_host_k = nullptr, *d_host_v = nullptr;
    ck(cudaHostAlloc(&h_k_pool, pool_bytes, cudaHostAllocMapped), "hostalloc k");
    ck(cudaHostAlloc(&h_v_pool, pool_bytes, cudaHostAllocMapped), "hostalloc v");
    ck(cudaHostGetDevicePointer((void**) &d_host_k, h_k_pool, 0), "hostptr k");
    ck(cudaHostGetDevicePointer((void**) &d_host_v, h_v_pool, 0), "hostptr v");
    ck(cudaMemset(h_k_pool, 0xFF, pool_bytes), "memset host k");
    ck(cudaMemset(h_v_pool, 0xFF, pool_bytes), "memset host v");
    uint8_t* d_stage_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_stage_v = dalloc<uint8_t>(pool_bytes);

    // Fresh VRAM pools for the batch test (the [1/4] pools keep their content for the gather test below).
    uint8_t* d_k2 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v2 = dalloc<uint8_t>(pool_bytes);
    k::KvHostPools host_pools;
    host_pools.k_turbo = d_host_k;
    host_pools.v_turbo = d_host_v;
    k::KvHostPools stage_pools;
    stage_pools.k_turbo = d_stage_k;
    stage_pools.v_turbo = d_stage_v;
    k::kv_append_turbo2(d_k2, d_v2, d_table, pos0, T, d_K, d_V, s, nullptr, &host_pools, &stage_pools);
    ck(cudaDeviceSynchronize(), "kv_append_turbo2");

    // steps form: the same cells, one launch, step_stride 4 (the mtp.cpp call shape), into a third pool set.
    uint8_t* d_k3 = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v3 = dalloc<uint8_t>(pool_bytes);
    std::vector<int32_t> steps_dev((size_t) T * k::kStepCount);
    for (int t = 0; t < T; ++t) {
        steps_dev[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) (pos0 + t);
        steps_dev[(size_t) t * k::kStepCount + k::kStepWidth] = 1;
    }
    int32_t* d_steps = dalloc<int32_t>(steps_dev.size());
    ck(cudaMemcpy(d_steps, steps_dev.data(), steps_dev.size() * 4, cudaMemcpyHostToDevice), "memcpy steps");
    // K/V laid out [T][H][D] is exactly what _steps reads with cur rows H*D apart.
    k::kv_append_turbo2_steps(d_k3, d_v3, d_table, d_steps, k::kStepCount, T, d_K, d_V, s, nullptr);
    ck(cudaDeviceSynchronize(), "kv_append_turbo2_steps");

    // planes=1 fold: v == k and vcur == kcur (the K8V4-style call shape) - one store per group.
    uint8_t* d_fold = dalloc<uint8_t>(pool_bytes);
    k::kv_append_turbo2_steps(d_fold, d_fold, d_table, d_steps, k::kStepCount, T, d_K, d_K, s, nullptr);
    ck(cudaDeviceSynchronize(), "kv_append_turbo2 fold");

    std::vector<uint8_t> h_k2(pool_bytes), h_v2(pool_bytes), h_k3(pool_bytes), h_v3(pool_bytes), h_fold(pool_bytes);
    ck(cudaMemcpy(h_k2.data(), d_k2, pool_bytes, cudaMemcpyDeviceToHost), "d2h k2");
    ck(cudaMemcpy(h_v2.data(), d_v2, pool_bytes, cudaMemcpyDeviceToHost), "d2h v2");
    ck(cudaMemcpy(h_k3.data(), d_k3, pool_bytes, cudaMemcpyDeviceToHost), "d2h k3");
    ck(cudaMemcpy(h_v3.data(), d_v3, pool_bytes, cudaMemcpyDeviceToHost), "d2h v3");
    ck(cudaMemcpy(h_fold.data(), d_fold, pool_bytes, cudaMemcpyDeviceToHost), "d2h fold");

    long bad2 = 0;
    for (int t = 0; t < T; ++t) {
        const int pos = (int) (pos0 + t);
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, pos, h, H, P);
                const float* raw = (is_v ? V : K).data() + ((size_t) t * H + h) * D;
                const uint8_t* pool_arr[2] = {(is_v ? h_v2.data() : h_k2.data()), (is_v ? h_v3.data() : h_k3.data())};
                for (int which = 0; which < 2; ++which) {
                    const uint8_t* pool_ptr = pool_arr[which] + row * bytes_per_head;
                    for (int b = 0; b < blocks_per_head; ++b) {
                        k::block_turbo2 ref;
                        quantize_block_turbo2_host(raw + b * 32, ref);
                        const k::block_turbo2* gpu = reinterpret_cast<const k::block_turbo2*>(pool_ptr) + b;
                        if (std::memcmp(&ref, gpu, sizeof(k::block_turbo2)) != 0) ++bad2;
                    }
                }
                // host copy: identity layout, row = ((pos/P)*H + h)*P + pos%P
                const long long hrow = (((long long) pos / P) * H + h) * P + (pos % P);
                const uint8_t* hp = (is_v ? h_v_pool : h_k_pool) + hrow * bytes_per_head;
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo2 ref;
                    quantize_block_turbo2_host(raw + b * 32, ref);
                    const k::block_turbo2* gpu = reinterpret_cast<const k::block_turbo2*>(hp) + b;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo2)) != 0) ++bad2;
                }
                // the fold pool holds K quantized for every cell (v folded onto k)
                for (int b = 0; b < blocks_per_head; ++b) {
                    k::block_turbo2 ref;
                    quantize_block_turbo2_host(K.data() + ((size_t) t * H + h) * D + b * 32, ref);
                    const k::block_turbo2* gpu = reinterpret_cast<const k::block_turbo2*>(h_fold.data() + row * bytes_per_head) + b;
                    if (std::memcmp(&ref, gpu, sizeof(k::block_turbo2)) != 0) ++bad2;
                }
            }
        }
    }
    // staging pool == VRAM pool (identity layout vs permuted: compare per logical cell)
    std::vector<uint8_t> h_sk(pool_bytes), h_sv(pool_bytes);
    ck(cudaMemcpy(h_sk.data(), d_stage_k, pool_bytes, cudaMemcpyDeviceToHost), "d2h stage k");
    ck(cudaMemcpy(h_sv.data(), d_stage_v, pool_bytes, cudaMemcpyDeviceToHost), "d2h stage v");
    for (int t = 0; t < T; ++t) {
        const int pos = (int) (pos0 + t);
        for (int h = 0; h < H; ++h) {
            const long long srow = (((long long) pos / P) * H + h) * P + (pos % P);
            for (int is_v = 0; is_v < 2; ++is_v) {
                const uint8_t* src = (is_v ? h_sv.data() : h_sk.data()) + srow * bytes_per_head;
                const uint8_t* dst = (is_v ? h_v2.data() : h_k2.data()) + row_of(table, pos, h, H, P) * bytes_per_head;
                if (std::memcmp(src, dst, bytes_per_head) != 0) ++bad2;
            }
        }
    }
    if (bad2) {
        std::fprintf(stderr, "FAIL: %ld mismatches in the batch / steps / host-copy / staging / fold checks\n", bad2);
        ++g_fail;
    } else {
        std::printf("  -> batch, steps, host copy, staging pool and the planes=1 fold all bitwise (OK)\n");
    }

    // -------------------------------------------------------------
    // [3/4] Gather: FP16 scratch bitwise vs a host dequant of the stored blocks
    // -------------------------------------------------------------
    std::printf("[3/4] kv_gather_turbo2_step: FP16 scratch bitwise vs host dequant...\n");
    const int max_ids = 96;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scr = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scr = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = positions[rng() % n_fill];   // shuffled across pages
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");
    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");

    k::kv_gather_turbo2_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, d_k_scr, d_v_scr, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_turbo2_step");

    std::vector<uint16_t> h_k_scr((size_t) max_ids * H * D), h_v_scr((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_k_scr.data(), d_k_scr, h_k_scr.size() * 2, cudaMemcpyDeviceToHost), "d2h k_scr");
    ck(cudaMemcpy(h_v_scr.data(), d_v_scr, h_v_scr.size() * 2, cudaMemcpyDeviceToHost), "d2h v_scr");

    long bad_gather = 0;
    double worst_err = 0.0, worst_rel = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, cell, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const uint16_t* got = (is_v ? h_v_scr.data() : h_k_scr.data()) + (size_t) (i * H + h) * D;
                const float* raw = (is_v ? host_v[cell] : host_k[cell]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    const k::block_turbo2* stored = reinterpret_cast<const k::block_turbo2*>(pool_ptr) + b;
                    uint16_t ref[32];
                    dequant_block_turbo2_host(*stored, ref);
                    if (std::memcmp(ref, got + b * 32, 64) != 0) ++bad_gather;
                    for (int t = 0; t < 32; ++t) {
                        const double err = std::fabs(k::f32_from_f16(got[b * 32 + t]) - raw[t + b * 32]);
                        if (err > worst_err) worst_err = err;
                        const double scale = std::fabs(k::f32_from_f16(stored->d));
                        if (scale > 1e-6 && err / scale > worst_rel) worst_rel = err / scale;
                    }
                }
            }
        }
    }
    if (bad_gather) {
        std::fprintf(stderr, "FAIL: %ld gathered values differ from the host dequant\n", bad_gather);
        ++g_fail;
    } else {
        std::printf("  -> gather bitwise vs host dequant (OK); max |x'-x| = %.4f, max relative to the block scale "
                    "= %.4f (centroids +-0.25*d apart)\n", worst_err, worst_rel);
    }

    // -------------------------------------------------------------
    // [4/4] Reconstruction sanity: 2-bit after the rotation stays bounded
    // -------------------------------------------------------------
    std::printf("[4/4] Reconstruction bound: every dequantized value within 0.75*d of its input...\n");
    double bound = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long row = row_of(table, cell, h, H, P);
                const uint8_t* pool_ptr = (is_v ? h_v.data() : h_k.data()) + row * bytes_per_head;
                const uint16_t* got = (is_v ? h_v_scr.data() : h_k_scr.data()) + (size_t) (i * H + h) * D;
                const float* raw = (is_v ? host_v[cell] : host_k[cell]).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    const float d = k::f32_from_f16(reinterpret_cast<const k::block_turbo2*>(pool_ptr)[b].d);
                    for (int t = 0; t < 32; ++t) {
                        const double err = std::fabs(k::f32_from_f16(got[b * 32 + t]) - raw[t + b * 32]);
                        // centroids are +-0.25*d and +-0.75*d; the worst rounding error is half the outer step
                        if (err > bound) bound = err;
                        if (d > 1e-6f && err > 0.625 * (double) d + 1e-3) {
                            std::fprintf(stderr, "FAIL: value %g dequantized to %g with scale %g\n",
                                         (double) raw[t + b * 32], (double) k::f32_from_f16(got[b * 32 + t]), (double) d);
                            ++g_fail;
                        }
                    }
                }
            }
        }
    }
    std::printf("  -> max reconstruction error %.4f (bound 0.625*d holds, OK)\n", bound);

    // -------------------------------------------------------------
    // turbo3: the same phases with the 3-bit pools (host reference vs CUDA, all bitwise)
    // -------------------------------------------------------------
    run_turbo3_phases(rng, nd);
    run_k3v8_phases(rng, nd);

    cudaFree(d_table); cudaFree(d_k); cudaFree(d_v); cudaFree(d_k2); cudaFree(d_v2); cudaFree(d_k3); cudaFree(d_v3);
    cudaFree(d_fold); cudaFree(d_stage_k); cudaFree(d_stage_v); cudaFree(d_K); cudaFree(d_V); cudaFree(d_steps);
    cudaFree(d_ids); cudaFree(d_k_scr); cudaFree(d_v_scr); cudaFree(d_kcur); cudaFree(d_vcur); cudaFree(d_step);
    ck(cudaFreeHost(h_k_pool), "free host k");
    ck(cudaFreeHost(h_v_pool), "free host v");

    if (g_fail == 0) {
        std::printf("=== kv_turbo_parity: ALL TESTS PASSED ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_turbo_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
