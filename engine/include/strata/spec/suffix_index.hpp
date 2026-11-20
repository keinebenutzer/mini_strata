// include/strata/spec/suffix_index.hpp - the session-wide suffix automaton drafter (no weights, no GPU).
//
// Ported from Kinetic (github.com/Veloresearch/kinetic, crates/kinetic-models/src/speculative/session.rs,
// Apache-2.0), the engine whose "Zero Wait" answers while the user is still typing. Same job as SuffixDrafter -
// propose the tokens that followed an earlier occurrence of the context's current suffix - but indexed by a
// suffix automaton over the whole history instead of a trigram table:
//
//   * the LONGEST earlier occurrence of ANY suffix is found in amortized O(1) per appended token (the trigram
//     index can only start from a 3-token key and extends backwards, capped at max_match; long repeats of
//     edited or quoted code match exactly here);
//   * every state keeps its two latest end positions, so a proposal continues from the latest occurrence that
//     is not the context itself (periodic text included);
//   * memory is bounded: past `capacity` tokens the index forgets the older half and rebuilds (~100 bytes per
//     held token), so a session that runs for hours cannot grow it.
//
// Matches shorter than `min_match` (counting the token about to run) propose nothing. Like every drafter, it
// can only change the speed: the verify window decides what is kept.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace strata::spec {

class SuffixIndex {
public:
    explicit SuffixIndex(size_t capacity = 1u << 18, int min_match = 3);

    void reset();
    /// Add tokens to the history. Past the capacity: forget the older half, rebuild, re-add the newer half.
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);
    /// As propose(), for the history followed by `extra` (drafts not yet verified): the suffix being matched
    /// runs through them, the proposal continues where their best earlier occurrence ends. The history is
    /// unchanged; `extra` is never indexed.
    int propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out);
    /// Match `query` (tokens not yet in the history) against the automaton, then take the latest occurrence that
    /// is not the context itself: writes up to `limit` continuation tokens, returns how many, sets last_match_.
    int draft(const int32_t* query, int n_query, int limit, int32_t* out);

    /// Length of the match behind the last proposal (0 if none), counting the token about to run.
    int last_match() const { return last_match_; }
    size_t size() const { return tokens_.size(); }
    const std::vector<int32_t>& history() const { return tokens_; }

private:
    static constexpr uint32_t kNone = 0xFFFFFFFFu;
    /// How far up the suffix-link chain a push refreshes the two latest end positions. Deeper states keep an
    /// older occurrence, which is still a real one.
    static constexpr int kRecencyDepth = 64;

    struct Node {
        uint32_t len = 0;           // longest string the state stands for
        uint32_t link = kNone;      // state of the longest suffix that ends in more places
        uint32_t first_out = kNone; // head of the state's edge list in out_
        uint32_t recent = kNone;    // latest end position of the state's strings
        uint32_t earlier = kNone;   // second latest
    };

    void push(int32_t c);
    uint32_t split(uint32_t p, uint32_t q, int32_t c);
    uint32_t new_node(uint32_t len, uint32_t link, uint32_t recent, uint32_t earlier);
    void add_edge(uint32_t from, int32_t token, uint32_t to);
    uint32_t edge(uint32_t from, int32_t token) const;
    const Node& node(uint32_t i) const { return nodes_[i]; }

    int min_match_;
    size_t capacity_;
    std::vector<int32_t> tokens_;
    std::vector<Node> nodes_;
    std::unordered_map<uint64_t, uint32_t> edges_;             // (state << 32 | token) -> target
    std::vector<std::pair<int32_t, uint32_t>> out_;           // per-state linked edge lists
    uint32_t last_ = 0;                                       // state of the whole history
    int last_match_ = 0;
};

}  // namespace strata::spec
