// include/strata/kernels/kv_k3v8.hpp - the asymmetric `--kv k3v8` mode: K as turbo3 (3-bit codes after the
// fwht256 rotation, kv_turbo3.hpp), V as plain int8 with an fp16 scale per 64 values (kv_q8.hpp), both halves
// on the SAME rotation contract. No kernels of its own: like kKvHybrid, it calls the two single-format
// kernels once per half with the unused half's lanes folded onto the used pool (kv_stream.hpp's
// kv_k3v8_*_half), so every write, host copy and gather is a call an existing kernel already performs.
//
// Why asymmetric: attention scores weight K through the softmax and tolerate coarse codes less than V, whose
// values average out - the llama.cpp lane runs -ctk turbo3 -ctv q8_0 for the same reason. K keeps the rotated
// 3-bit form (112 B per cell and head), V keeps 8-bit accuracy (head_dim codes + head_dim/64 fp16 scales).
//
// Bytes per cell (n_head_kv heads, K + V): n_head_kv * (112 + head_dim + head_dim/64*2) = 752 at head_dim 256
// and 2 heads (turbo3 alone: 448, k8v4: 816, q4_0: 576, int8: 1056).
#ifndef STRATA_KERNELS_KV_K3V8_HPP
#define STRATA_KERNELS_KV_K3V8_HPP

#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_turbo3.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// One cell side (K or V) of one head: the turbo3 K half plus the int8 V half, per head.
inline uint64_t kv_k3v8_bytes_per_head(int head_dim) {
    return kv_turbo3_bytes_per_head(head_dim) + (uint64_t) head_dim +
           (uint64_t) (head_dim / KV_Q8_GROUP) * 2;
}

/// One KV cell (both sides, all KV heads) in bytes: every head carries its K half AND its V half, so the
/// per-head sum IS the cell (unlike the single-format getters, no factor 2).
inline uint64_t kv_k3v8_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_k3v8_bytes_per_head((int) s.head_dim);
}

}  // namespace strata::kernels

#endif  // STRATA_KERNELS_KV_K3V8_HPP
