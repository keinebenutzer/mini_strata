// include/strata/kernels/kv_turbo3.hpp - 3-bit KV storage on kv_q4.hpp's Walsh-Hadamard rotation (`--kv turbo3`).
//
// turbo3 rides the SAME rotation sites as q4_0 and turbo2 (layer.cpp's kv_rot branches: K and V rotate through
// fwht256_inplace_cuda before the append, the query before attention, the attention output back) and replaces
// only the quantizer and the storage. It is turbo2's structure with one more bit: 8 uniform levels (midtreads)
// instead of 4, 14 bytes per 32-value block instead of 10.
//
// Quantization is q4_0-style max-abs scaling over blocks of 32 values, with the scale chosen so the largest
// magnitude in the block lands exactly on the outer centroid:
//     amax = max|x| over the 32 values
//     d    = fp16(amax / 0.875)                       - the stored scale
//     hd   = d * 0.25                                 - the centroid spacing (exact: * 0.25)
//     code = clamp(rint(x / hd + 3.5), 0, 7)          - so amax -> code 7
//     x'   = fp16((code - 3.5) * hd)                  - centroids {-0.875, -0.625, ..., +0.875} * d
// (The CPU reference and the CUDA kernel run the identical float expressions - IEEE division, rintf's
// round-half-to-even, and f16_bits.hpp's converter - so they agree bit for bit; kv_turbo_parity checks that.)
//
// Layout per 32 values: an fp16 scale + 12 bytes of 3-bit codes = 14 bytes per block, 112 B per head per cell
// for head_dim 256 - vs 80 in turbo2, 144 in q4_0, 264 in INT8 and 512 in FP16. Per cell (K and V of both KV
// heads): 448 B, vs 320 in turbo2 and 576 in q4_0.
//
// The codes are a compact little-endian bitstream: 32 values x 3 bits = 96 bits, value i occupies bits
// 3i..3i+2 of the 12-byte array, crossing byte boundaries (byte j holds bits 8j..8j+7). The CPU reference and
// the CUDA kernel pack and unpack the identical bitstream - kv_turbo_parity checks that bitwise.
//
// The gather dequantizes into the same FP16 scratch as q4_0's and the values stay ROTATED (the inverse is the
// caller's, on the attention output); qsa_decode_attn also reads these pools directly (its KV_MODE 5).
// With KV streaming the codes live in the host copy like turbo2's (KvHostPools k_turbo3/v_turbo3, same granule).
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int KV_TURBO3_GROUP = 32;

#pragma pack(push, 1)
struct block_turbo3 {
    uint16_t d;         // scale (fp16 bits): amax / 0.875, so amax lands on the outer centroid 0.875*d
    uint8_t qs[KV_TURBO3_GROUP * 3 / 8];   // 32 3-bit codes as a little-endian bitstream: value i in bits 3i..3i+2
};
#pragma pack(pop)

static_assert(sizeof(block_turbo3) == 14, "block_turbo3 must be 14 bytes");

/// Bytes per cell and KV head: 8 blocks of 32 for head_dim 256 -> 112.
inline uint64_t kv_turbo3_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / KV_TURBO3_GROUP) * sizeof(block_turbo3); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_turbo3_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_turbo3_bytes_per_head((int) s.head_dim) * 2;
}

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_turbo3_step(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, const int32_t* step,
                           const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                           const KvHostPools* host = nullptr);

void kv_append_turbo3_steps(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, const int32_t* step,
                            int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                            void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_turbo3(uint8_t* k_turbo3, uint8_t* v_turbo3, const int32_t* page_table, int64_t pos0, int64_t T,
                      const float* K, const float* V, const QsaShapes& s, void* stream,
                      const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_turbo3_step(const uint8_t* k_turbo3, const uint8_t* v_turbo3, const int32_t* page_table,
                           const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                           uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
