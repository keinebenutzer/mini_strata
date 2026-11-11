// include/strata/spec/draft_source.hpp - a source of draft tokens beside the MTP layer (--lookup-chain).
//
// A source proposes how the context continues; the verify window checks the proposal, so a source can only change
// the speed, never the output. The engine asks its sources after the MTP has drafted: the context it passes ends
// with those MTP drafts (not yet verified), and whatever a source returns is appended to the same window.
//
// The built-in source is the request's own text (PromptLookupSource: the suffix drafter over prompt + output).
// Other sources (a retrieval store over a codebase or earlier conversations) register with
// extra_draft_sources() before the first request; the engine takes the proposal with the longest match.
#pragma once

#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/suffix_index.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::spec {

class DraftSource {
public:
    virtual ~DraftSource() = default;
    virtual const char* name() const = 0;
    /// A new request starts (its prompt follows through append()).
    virtual void reset() {}
    /// Committed context, in order: the request's prompt, then every token the verify windows accept.
    virtual void append(const int32_t* tokens, size_t n) { (void) tokens; (void) n; }
    /// Propose up to `max_k` tokens that follow the context. `tail` holds the context's last `n_tail` tokens (at most
    /// 64); its last `n_pending` are drafts not yet verified (never passed to append()). Writes `out`, returns how
    /// many (0 = nothing to propose) and sets `*match` to the number of trailing context tokens the proposal is based
    /// on (its confidence: the engine requires --lookup-chain-min and prefers the longest).
    virtual int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) = 0;
};

/// The request's own text: the suffix drafter (prompt lookup) behind the DraftSource interface.
class PromptLookupSource final : public DraftSource {
public:
    explicit PromptLookupSource(SuffixDrafter& d) : d_(d) {}
    const char* name() const override { return "prompt-lookup"; }
    void reset() override { d_.reset(); }
    void append(const int32_t* tokens, size_t n) override { d_.append(tokens, n); }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        n_pending = n_pending < n_tail ? n_pending : n_tail;
        const int k = d_.propose_after(tail + (n_tail - n_pending), n_pending, max_k, out);
        *match = k > 0 ? d_.last_match() : 0;
        return k;
    }

private:
    SuffixDrafter& d_;
};

/// The whole session (all requests since the engine started), indexed by the suffix automaton: a proposal may
/// continue text an earlier request wrote, and matches longer than the trigram drafter's 64-token cap.
class SessionLookupSource final : public DraftSource {
public:
    explicit SessionLookupSource(SuffixIndex& idx) : idx_(idx) {}
    const char* name() const override { return "session-automaton"; }
    /// The session keeps what it learned; a new request only extends it.
    void reset() override {}
    /// Kinetic's lag (speculative/mod.rs): the newest committed token is pushed onto the context before the
    /// drafter proposes and only observed once the next token is known - the automaton holds everything
    /// except the token whose continuation we predict. We hold that one token back until the next append.
    void append(const int32_t* tokens, size_t n) override {
        if (n == 0) return;
        if (have_held_) idx_.append(&held_, 1);
        held_ = tokens[n - 1];
        have_held_ = true;
        idx_.append(tokens, n - 1);
    }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        if (!have_held_) return 0;
        n_pending = n_pending < n_tail ? n_pending : n_tail;
        int32_t query[64];
        query[0] = held_;
        for (int i = 0; i < n_pending; ++i) query[1 + i] = tail[n_tail - n_pending + i];
        const int k = idx_.draft(query, n_pending + 1, max_k, out);
        *match = k > 0 ? idx_.last_match() : 0;
        return k;
    }

private:
    SuffixIndex& idx_;
    int32_t held_ = 0;
    bool have_held_ = false;
};

/// Sources besides the request's own text, asked after it (empty unless something registered one at startup).
std::vector<std::unique_ptr<DraftSource>>& extra_draft_sources();

/// STRATA_LOOKUP_SESSION=1: register the built-in session automaton (Kinetic's cross-request lookup, see
/// strata/spec/suffix_index.hpp). Call once at engine start; without the variable it registers nothing, and a
/// second call is a no-op. The index lives for the process and keeps what it learned across requests - unlike
/// the request's own drafter, which reset() clears.
void ensure_session_lookup_source();

/// Ask `own` and then every extra source; the proposal with the longest match of at least `min_match` wins
/// (ties keep the earlier source). Returns its length (0 = none); `*match` and `*source` (index: 0 = own) describe it.
int propose_from_sources(DraftSource& own, const int32_t* tail, int n_tail, int n_pending, int max_k, int min_match,
                         int32_t* out, int* match, int* source);

}  // namespace strata::spec
