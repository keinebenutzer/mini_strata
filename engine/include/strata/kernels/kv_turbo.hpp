// include/strata/kernels/kv_turbo.hpp - 2-bit KV storage on kv_q4.hpp's Walsh-Hadamard rotation (`--kv turbo2`).
//
// turbo2 rides the SAME rotation sites as q4_0 (layer.cpp's kv_rot branches: K and V rotate through
// fwht256_inplace_cuda before the append, the query before attention, the attention output back) and replaces
// only the quantizer and the storage. After the rotation a head's outliers are spread over all 256 dimensions,
// which is what makes 2 bits worth trying at all; what is lost to 2 bits is measured, not assumed.
//
// Quantization is q4_0-style max-abs scaling over blocks of 32 values, with the scale chosen so the largest
// magnitude in the block lands exactly on the outer centroid:
//     amax = max|x| over the 32 values
//     d    = fp16(amax / 0.75)                       - the stored scale
//     code = clamp(rint(x / (d / 2) + 1.5), 0, 3)    - d/2 is the centroid spacing, so amax -> code 3
//     x'   = fp16((code - 1.5) * (d / 2))            - centroids {-0.75, -0.25, +0.25, +0.75} * d
// (The task's `x/d` form used d for the spacing; here d is the stored scale and the spacing is d/2, which is
// what makes amax map to 0.75*d exactly. The CPU reference and the CUDA kernel run the identical float
// expressions - IEEE division, rintf's round-half-to-even, and f16_bits.hpp's converter - so they agree bit
// for bit; kv_turbo_parity checks that.)
//
// Layout per 32 values: an fp16 scale + 8 bytes of 2-bit codes (4 codes per byte, element j in bits
// 2*(j%4)..2*(j%4)+1 of qs[j/4]) = 10 bytes per block, 80 B per head per cell for head_dim 256 - vs 144 in
// q4_0, 264 in INT8 and 512 in FP16. Per cell (K and V of both KV heads): 320 B, vs 576 in q4_0.
//
// The gather dequantizes into the same FP16 scratch as q4_0's and the values stay ROTATED (the inverse is the
// caller's, on the attention output); qsa_decode_attn also reads these pools directly (its KV_MODE 4).
// With KV streaming the codes live in the host copy like q4_0's (KvHostPools k_turbo/v_turbo, same granule).
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int KV_TURBO_GROUP = 32;

#pragma pack(push, 1)
struct block_turbo2 {
    uint16_t d;         // scale (fp16 bits): amax / 0.75, so amax lands on the outer centroid 0.75*d
    uint8_t qs[KV_TURBO_GROUP / 4];   // 32 2-bit codes: element j in bits 2*(j%4).. of qs[j/4], low bits first
};
#pragma pack(pop)

static_assert(sizeof(block_turbo2) == 10, "block_turbo2 must be 10 bytes");

/// Bytes per cell and KV head: 8 blocks of 32 for head_dim 256 -> 80.
inline uint64_t kv_turbo_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / KV_TURBO_GROUP) * sizeof(block_turbo2); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_turbo_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_turbo_bytes_per_head((int) s.head_dim) * 2;
}

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_turbo2_step(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                           const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                           const KvHostPools* host = nullptr);

void kv_append_turbo2_steps(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, const int32_t* step,
                            int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                            void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_turbo2(uint8_t* k_turbo, uint8_t* v_turbo, const int32_t* page_table, int64_t pos0, int64_t T,
                      const float* K, const float* V, const QsaShapes& s, void* stream,
                      const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_turbo2_step(const uint8_t* k_turbo, const uint8_t* v_turbo, const int32_t* page_table,
                           const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                           uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
