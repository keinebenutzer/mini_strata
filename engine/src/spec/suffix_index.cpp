// src/spec/suffix_index.cpp - see include/strata/spec/suffix_index.hpp.
//
// The automaton is the classic suffix automaton (endpos classes, suffix links); the two latest end positions per
// state are refreshed along the suffix-link chain of every pushed token, up to kRecencyDepth states. A proposal
// runs matching statistics: the longest suffix of (history + extra) that is a substring of the history, then the
// continuation after its latest occurrence that is not the context itself.
#include "strata/spec/suffix_index.hpp"

#include <algorithm>

namespace strata::spec {

namespace {
uint64_t edge_key(uint32_t from, int32_t token) {
    return ((uint64_t) from << 32) | (uint32_t) token;
}
}  // namespace

SuffixIndex::SuffixIndex(size_t capacity, int min_match)
    : min_match_(std::max(1, min_match)), capacity_(std::max<size_t>(2, capacity)) {
    nodes_.reserve(capacity_ * 2);
    edges_.reserve(capacity_ * 3);
    out_.reserve(capacity_ * 3);
    tokens_.reserve(capacity_);
    nodes_.push_back(Node{});   // the root: state 0
}

void SuffixIndex::reset() {
    tokens_.clear();
    nodes_.clear();
    edges_.clear();
    out_.clear();
    nodes_.push_back(Node{});
    last_ = 0;
    last_match_ = 0;
}

uint32_t SuffixIndex::new_node(uint32_t len, uint32_t link, uint32_t recent, uint32_t earlier) {
    nodes_.push_back(Node{len, link, kNone, recent, earlier});
    return (uint32_t) nodes_.size() - 1;
}

uint32_t SuffixIndex::edge(uint32_t from, int32_t token) const {
    const auto it = edges_.find(edge_key(from, token));
    return it == edges_.end() ? kNone : it->second;
}

void SuffixIndex::add_edge(uint32_t from, int32_t token, uint32_t to) {
    edges_[edge_key(from, token)] = to;
    const uint32_t prev_head = nodes_[from].first_out;   // read before the entry is appended (it may reallocate)
    nodes_[from].first_out = (uint32_t) out_.size();
    out_.push_back({token, prev_head});
}

void SuffixIndex::push(int32_t c) {
    const uint32_t end = (uint32_t) tokens_.size();
    tokens_.push_back(c);
    const uint32_t cur = new_node(node(last_).len + 1, kNone, end, kNone);
    uint32_t p = last_;
    while (p != kNone && edge(p, c) == kNone) {
        add_edge(p, c, cur);
        p = node(p).link;
    }
    uint32_t link;
    if (p == kNone) {
        link = 0;
    } else {
        const uint32_t q = edge(p, c);
        if (node(p).len + 1 == node(q).len) {
            link = q;
        } else {
            link = split(p, q, c);
        }
    }
    nodes_[cur].link = link;
    last_ = cur;

    uint32_t s = link;
    for (int i = 0; i < kRecencyDepth; ++i) {
        if (s == 0) break;
        nodes_[s].earlier = nodes_[s].recent;
        nodes_[s].recent = end;
        s = node(s).link;
    }
}

uint32_t SuffixIndex::split(uint32_t p, uint32_t q, int32_t c) {
    const uint32_t link = node(q).link, recent = node(q).recent, earlier = node(q).earlier;
    const uint32_t clone = new_node(node(p).len + 1, link, recent, earlier);
    for (uint32_t e = node(q).first_out; e != kNone; e = out_[e].second)
        add_edge(clone, out_[e].first, edges_.at(edge_key(q, out_[e].first)));
    while (p != kNone && edge(p, c) == q) {
        edges_[edge_key(p, c)] = clone;
        p = node(p).link;
    }
    nodes_[q].link = clone;
    return clone;
}

void SuffixIndex::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (tokens_.size() == capacity_) {   // forget the older half, rebuild with the newer half
            std::vector<int32_t> keep(tokens_.begin() + (long) (capacity_ / 2), tokens_.end());
            reset();
            for (int32_t t : keep) push(t);
        }
        push(tokens[i]);
    }
}

int SuffixIndex::draft(const int32_t* query, int n_query, int limit, int32_t* out) {
    last_match_ = 0;
    const size_t n = tokens_.size();
    if (n == 0 || limit <= 0) return 0;

    // Matching statistics over `query`, starting from the whole history (which matches itself): the longest
    // suffix of (history + query) that is a substring of the history. Descending suffix links is Kinetic's
    // walk from `last` looking for the transition on the token about to run - the same states, the same match.
    uint32_t v = last_;
    size_t l = node(last_).len;
    for (int i = 0; i < n_query && l > 0; ++i) {
        const int32_t c = query[i];
        uint32_t to = edge(v, c);
        if (to != kNone) { v = to; ++l; continue; }
        while (v != kNone && (to = edge(v, c)) == kNone) v = node(v).link;
        if (v == kNone) { v = 0; l = 0; continue; }   // the token is not in the language at all
        l = node(v).len + 1;
        v = to;
    }
    if ((int) l < min_match_) return 0;

    // The continuation after the latest occurrence that is not the context itself (recent may be the history's
    // own end; earlier is the second latest, and an older occurrence is still a real one). The read-back stops
    // at the history's end - what follows there is the context, not a remembered continuation.
    const Node& found = node(v);
    const uint32_t ends[2] = {found.recent, found.earlier};
    for (const uint32_t end : ends) {
        if (end == kNone) continue;
        const size_t start = (size_t) end + 1;
        if (start >= n) continue;
        last_match_ = (int) l;
        int k = 0;
        const size_t stop = std::min(start + (size_t) limit, n);
        for (size_t q = start; q < stop; ++q) out[k++] = tokens_[q];
        return k;
    }
    return 0;
}

int SuffixIndex::propose(int max_k, int32_t* out) {
    if (tokens_.empty()) return 0;
    const int32_t next = tokens_.back();   // the token about to run is part of the matched suffix
    return draft(&next, 1, max_k, out);
}

int SuffixIndex::propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    if (n_extra <= 0) return propose(max_k, out);
    return draft(extra, n_extra, max_k, out);   // the extra tokens extend the matched suffix; they are never indexed
}

}  // namespace strata::spec
