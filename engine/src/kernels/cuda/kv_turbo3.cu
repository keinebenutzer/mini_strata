// src/kernels/cuda/kv_turbo3.cu - see include/strata/kernels/kv_turbo3.hpp. 3-bit KV on kv_q4.hpp's rotation.
// The kernels mirror kv_turbo.cu one for one (same grids, same device-side page/step indirection, same host-copy
// and staging writes); only the group quantizer and the block layout differ (8 levels, 14-byte blocks, a 3-bit
// little-endian bitstream instead of 4 packed 2-bit bytes).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_turbo3.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_turbo3: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One 32-value group in one warp (lane = element): d = fp16(amax / 0.875) so amax lands on the outer centroid,
// q = clamp(rint(x / (d*0.25) + 3.5), 0, 7). Returns the scale bits; `bytes[0..11]` is the group's 96-bit
// little-endian code bitstream (value i in bits 3i..3i+2) replicated to every lane. The scale depends on |x|
// only, so a plain shuffle max of fabsf is deterministic in every lane - no sign tie-break needed (as turbo2).
// The packing is a warp-wide OR of each lane's disjoint 3-bit field qc << 3*lane, held in a 128-bit accumulator
// (the fields never overlap, so OR is associative and every lane ends with the identical 96-bit stream - the
// same stream the host reference builds byte by byte).
__device__ __forceinline__ uint16_t turbo3_group(float x, int lane, uint8_t* bytes) {
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const uint16_t dbits = f16_from_f32(amax / 0.875f);          // the stored scale, rounded once
    const float hd = f32_from_f16(dbits) * 0.25f;                // the centroid spacing (exact: * 0.25)
    const float t = hd != 0.0f ? x / hd : 0.0f;                  // an all-zero block codes to 4 (dequant +0)
    const int q = (int) rintf(t + 3.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 7 ? 7 : q));
    unsigned __int128 acc = (unsigned __int128) qc << (3 * lane);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long lo = (unsigned long long) acc, hi = (unsigned long long) (acc >> 64);
        lo |= __shfl_xor_sync(0xffffffffu, lo, o);
        hi |= __shfl_xor_sync(0xffffffffu, hi, o);
        acc = ((unsigned __int128) hi << 64) | (unsigned __int128) lo;
    }
    const unsigned long long lo = (unsigned long long) acc, hi = (unsigned long long) (acc >> 64);
#pragma unroll
    for (int j = 0; j < 12; ++j)
        bytes[j] = j < 8 ? (uint8_t) (lo >> (8 * j)) : (uint8_t) (hi >> (8 * (j - 8)));
    return dbits;
}

__device__ __forceinline__ void turbo3_store(uint8_t* pool, long long row, int b, int lane, uint16_t d,
                                             const uint8_t* bytes) {
    block_turbo3* blk = reinterpret_cast<block_turbo3*>(pool + row * (long long) sizeof(block_turbo3) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 12) blk->qs[lane] = bytes[lane];
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__global__ void kv_append_turbo3_kernel(uint8_t* __restrict__ k_turbo3, uint8_t* __restrict__ v_turbo3,
                                        const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                        int step_stride, int planes,
                                        const float* __restrict__ kcur, const float* __restrict__ vcur,
                                        int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * KV_TURBO3_GROUP + t];
    uint8_t bytes[12];
    const uint16_t d = turbo3_group(x, t, bytes);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0)
        turbo3_store(is_v ? v_turbo3 : k_turbo3, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, bytes);
    if (host.k_turbo3 != nullptr)
        turbo3_store(is_v ? host.v_turbo3 : host.k_turbo3, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size),
                     b, t, d, bytes);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
__global__ void kv_append_turbo3_batch_kernel(uint8_t* __restrict__ k_turbo3, uint8_t* __restrict__ v_turbo3,
                                              const int32_t* __restrict__ table, int64_t pos0,
                                              const float* __restrict__ K, const float* __restrict__ V,
                                              int kv_heads, int head_dim, int page_size, int is_v_grid,
                                              KvHostPools host, KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * KV_TURBO3_GROUP + th];
    uint8_t bytes[12];
    const uint16_t d = turbo3_group(x, th, bytes);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0)
        turbo3_store(is_v ? v_turbo3 : k_turbo3, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, bytes);
    if (host.k_turbo3 != nullptr) turbo3_store(is_v ? host.v_turbo3 : host.k_turbo3, row_id, b, th, d, bytes);
    if (stage.k_turbo3 != nullptr) turbo3_store(is_v ? stage.v_turbo3 : stage.k_turbo3, row_id, b, th, d, bytes);
}

// The 3-bit unpack of value `e` of a group's bitstream (byte (3e)>>3, shift (3e)&7; shifts 6 and 7 straddle the
// next byte). The host reference (kv_turbo_parity.cpp) unpacks the same way.
__device__ __forceinline__ int turbo3_code(const block_turbo3* blk, int e) {
    const int by = (3 * e) >> 3, sh = (3 * e) & 7;
    if (sh <= 5) return (blk->qs[by] >> sh) & 7;
    return (((int) blk->qs[by] >> sh) | ((int) blk->qs[by + 1] << (8 - sh))) & 7;
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths); the values stay rotated.
__global__ void kv_gather_turbo3_kernel(const uint8_t* __restrict__ k_turbo3, const uint8_t* __restrict__ v_turbo3,
                                        const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                        const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                        uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / KV_TURBO3_GROUP;                    // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_turbo3);         // 112
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_turbo3* k_blk = reinterpret_cast<const block_turbo3*>(k_turbo3 + row * bytes_per_head) + b;
    const block_turbo3* v_blk = reinterpret_cast<const block_turbo3*>(v_turbo3 + row * bytes_per_head) + b;
    const float khd = f32_from_f16(k_blk->d) * 0.25f;
    const float vhd = f32_from_f16(v_blk->d) * 0.25f;
    const int kq = turbo3_code(k_blk, t);
    const int vq = turbo3_code(v_blk, t);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * KV_TURBO3_GROUP + t);
    k_scratch[dst_offset] = f16_from_f32(((float) kq - 3.5f) * khd);
    v_scratch[dst_offset] = f16_from_f32(((float) vq - 3.5f) * vhd);
}

void need_256_turbo3(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void kv_append_turbo3_steps(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, const int32_t* step,
                            int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                            void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256_turbo3(s, "kv_append_turbo3");
    const int planes = (v_turbo3 == nullptr || (v_turbo3 == k_turbo3 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_TURBO3_GROUP), (unsigned) (n_steps * planes));
    kv_append_turbo3_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_turbo3, v_turbo3, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_turbo3 launch");
}

void kv_append_turbo3_step(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, const int32_t* step,
                           const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                           const KvHostPools* host) {
    kv_append_turbo3_steps(k_turbo3, v_turbo3, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_turbo3(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, int64_t pos0, int64_t T,
                      const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                      const KvHostPools* stage) {
    if (T <= 0) return;
    need_256_turbo3(s, "kv_append_turbo3");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / KV_TURBO3_GROUP));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_turbo3_batch_kernel<<<grid, 32, 0, cs>>>(k_turbo3, v_turbo3, page_table, pos0, K, V,
                                                          (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
                                                          is_v, h, st);
    check("kv_append_turbo3 batch launch");
}

void kv_gather_turbo3_step(const uint8_t* k_turbo3, const uint8_t* v_turbo3, const int32_t* page_table,
                           const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                           uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / KV_TURBO3_GROUP);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_turbo3_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_turbo3, v_turbo3, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_turbo3 launch");
}

}  // namespace strata::kernels
