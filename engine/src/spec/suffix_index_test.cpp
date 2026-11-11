// src/spec/suffix_index_test.cpp - the session automaton drafter (kinetic port): unit tests and a simulation.
//
//   suffix_index_test                        unit tests
//   suffix_index_test --simulate [--k K] FILE.ids ...
//
// The simulation mirrors suffix_drafter_test's: replay text as if a model had produced it, draft up to K tokens,
// accept the matching prefix, commit it plus one bonus token. It measures what the automaton drafter can do on
// the same text the trigram drafter was measured on - longer repeats should pay more here.
#include "strata/spec/suffix_index.hpp"
#include "strata/spec/draft_source.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::SuffixIndex;
using strata::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

// Brute force (ported from Kinetic's reference test): the longest suffix of (history + [next]) that occurs in
// history, then the continuation after its latest occurrence that still has a token to read.
std::vector<int32_t> expected(const std::vector<int32_t>& history, int32_t next, int limit, int min_match) {
    std::vector<int32_t> text = history;
    text.push_back(next);
    const size_t n = history.size();
    for (size_t len = text.size(); len >= 1; --len) {
        std::vector<size_t> starts;                       // latest first
        for (size_t i = n >= len ? n - len + 1 : 0; i-- > 0;)
            if (std::equal(text.end() - (long) len, text.end(), history.begin() + (long) i))
                starts.push_back(i + len);
        if (starts.empty()) continue;
        if ((int) len < min_match) return {};
        for (size_t start : starts)
            if (start < n) return std::vector<int32_t>(history.begin() + (long) start,
                                                       history.begin() + (long) std::min(start + (size_t) limit, n));
        return {};
    }
    return {};
}

void unit_tests() {
    // 1. Kinetic's reference test 'continues_the_longest_and_latest_match': drafts extend the matched suffix
    {
        SuffixIndex d(1u << 12, 2);
        const int32_t h[] = {1, 2, 3, 4, 9, 9, 2, 3, 5, 7, 1, 2};
        d.append(h, 12);
        const int32_t pending[] = {3};
        int32_t out[8] = {};
        const int k = d.propose_after(pending, 1, 3, out);
        const int32_t want[] = {4, 9, 9};
        check(k == 3 && std::equal(out, out + 3, want), "kinetic: draft(3) == [4,9,9]");
        check(d.last_match() == 3, "kinetic: match counts the running token");
        const int32_t more[] = {3, 8, 8, 2};
        d.append(more, 4);
        const int k2 = d.propose_after(pending, 1, 2, out);
        const int32_t want2[] = {8, 8};
        check(k2 == 2 && std::equal(out, out + 2, want2), "kinetic: draft(3) == [8,8] after extend");
    }
    // 2. min_match gate: a 2-token match proposes nothing when 3 are required; unknown tokens never match
    {
        SuffixIndex d(1u << 12, 3);
        const int32_t h[] = {7, 8, 7, 8};
        d.append(h, 4);
        int32_t out[8] = {};
        check(d.propose(4, out) == 0, "gate: 2-token match under min_match 3");
        const int32_t pending[] = {9};   // 9 is not in the language at all
        check(d.propose_after(pending, 1, 4, out) == 0, "gate: unknown token");
    }
    // 3. periodic text through the session source (kinetic's lag: the automaton holds everything except the
    //    token about to run): after [1,2,3]x3 the suffix "1 2 3 1 2" occurred at 0-4 and 3-7, and 3 followed
    //    it at 0-5 - the proposal continues after 3-7 with [1,2]
    {
        SuffixIndex sa(1u << 12, 2);
        strata::spec::SessionLookupSource s(sa);
        const int32_t h[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        s.append(h, 9);
        int32_t out[8] = {};
        int match = 0;
        const int k = s.propose(h, 9, 0, 4, out, &match);
        const int32_t want[] = {1, 2};
        check(k == 2 && std::equal(out, out + 2, want), "periodic: continues after '1 2 3 1 2 3'");
        check(match == 6, "periodic: match length");
        // [1,2,3,4]x2+[1]: "1 2 3 4 1" occurred at 0-4, continuation [2,3,4]
        SuffixIndex sb(1u << 12, 2);
        strata::spec::SessionLookupSource s2(sb);
        const int32_t h2[] = {1, 2, 3, 4, 1, 2, 3, 4, 1};
        s2.append(h2, 9);
        const int k2 = s2.propose(h2, 9, 0, 4, out, &match);
        const int32_t want2[] = {2, 3, 4};
        check(k2 == 3 && std::equal(out, out + 3, want2), "periodic: continues after '1 2 3 4 1'");
        check(match == 5, "periodic: match length 5");
    }
    // 4. Kinetic's reference test 'forgets_the_older_half_when_full'
    {
        SuffixIndex d(8, 2);
        const int32_t h[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        d.append(h, 9);
        check(d.size() == 5, "capacity: older half forgotten (len 5)");
        const int32_t two[] = {2};
        int32_t out[8] = {};
        check(d.propose_after(two, 1, 2, out) == 0, "capacity: forgotten token no longer matches");
        d.append(5);
        const int32_t six[] = {6};
        const int k = d.propose_after(six, 1, 2, out);
        const int32_t want[] = {7, 8};
        check(k == 2 && std::equal(out, out + 2, want), "capacity: kept half still drafts");
    }
    // 5. brute-force parity on random short-alphabet text (Kinetic's 'drafts_like_brute_force')
    {
        for (int alphabet : {2, 3, 5}) {
            SuffixIndex d(1u << 12, 2);
            std::vector<int32_t> history;
            unsigned seed = 12345u + (unsigned) alphabet;
            auto rnd = [&seed](int m) { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) % (unsigned) m); };
            for (int step = 0; step < 400; ++step) {
                const int32_t t = (int32_t) rnd(alphabet);
                d.append(t);
                history.push_back(t);
                const int32_t next = (int32_t) rnd(alphabet);
                int32_t out[6] = {};
                const int k = d.propose_after(&next, 1, 6, out);
                const std::vector<int32_t> want = expected(history, next, 6, 2);
                check(k == (int) want.size() && std::equal(out, out + k, want.begin()), "parity: draft == brute force");
                if (g_fail) return;
            }
        }
    }
    // 6. brute-force parity for the production path through the session source (lag: the automaton holds
    //    everything except the token about to run, exactly like Kinetic's Decoder)
    {
        SuffixIndex d(1u << 12, 2);
        strata::spec::SessionLookupSource s(d);
        std::vector<int32_t> history;
        unsigned seed = 999u;
        auto rnd = [&seed](int m) { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) % (unsigned) m); };
        for (int step = 0; step < 400; ++step) {
            const int32_t t = (int32_t) rnd(4);
            s.append(&t, 1);
            history.push_back(t);
            int32_t out[6] = {};
            int match = 0;
            const int k = s.propose(history.data(), (int) history.size(), 0, 6, out, &match);
            const std::vector<int32_t> auto_hist(history.begin(), history.end() - 1);
            const std::vector<int32_t> want = expected(auto_hist, history.back(), 6, 2);
            check(k == (int) want.size() && std::equal(out, out + k, want.begin()), "parity: session propose == brute force");
            if (g_fail) return;
        }
    }
    // 7. the DraftSource wiring: the session source keeps learning across requests and wins on the longer match
    {
        SuffixIndex session(1u << 12, 3);
        strata::spec::extra_draft_sources().push_back(
            std::make_unique<strata::spec::SessionLookupSource>(session));   // as ensure_session_lookup_source does
        auto& src = *static_cast<strata::spec::SessionLookupSource*>(strata::spec::extra_draft_sources().back().get());
        SuffixDrafter fresh(3, 64, 64);   // a fresh per-request drafter: it knows only this request's prompt
        strata::spec::PromptLookupSource own(fresh);
        const int32_t r1[] = {10, 11, 12, 13, 14, 15};
        src.append(r1, 6);                       // request 1 wrote this
        const int32_t r2[] = {10, 11, 12};
        src.append(r2, 3);                       // request 2 quotes request 1's start
        own.append(r2, 3);
        // request 2's MTP drafts 13, 14 (pending): the session knows what followed (15) and continues past it
        const int32_t tail[] = {10, 11, 12, 13, 14};
        int32_t out[8] = {};
        int match = 0, source = -1;
        const int k = strata::spec::propose_from_sources(own, tail, 5, 2, 8, 3, out, &match, &source);
        strata::spec::extra_draft_sources().clear();
        check(k == 3 && match == 5 && source == 1, "source: session wins with the longer match");
        const int32_t want[] = {15, 10, 11};   // automaton = committed minus the held 12: "10 11 12 13 14" at 0-4
        check(k == 3 && std::equal(out, out + 3, want), "source: continues request 1's text");
    }
    // 8. speed smoke: 100k tokens appended, 1000 three-token lookups - the automaton must stay O(1) per step
    {
        SuffixIndex d(1u << 18, 3);
        std::vector<int32_t> text;
        unsigned seed = 7u;
        auto rnd = [&seed](int m) { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) % (unsigned) m); };
        for (int i = 0; i < 100000; ++i) text.push_back((int32_t) rnd(100));
        for (size_t i = 0; i + 40 <= text.size(); i += 100) text.insert(text.end(), text.begin() + (long) i, text.begin() + (long) i + 40);
        d.append(text.data(), text.size());
        int32_t out[8] = {};
        int hits = 0;
        for (int j = 0; j < 1000; ++j) hits += d.propose_after(&text[(size_t) j], 3, 8, out) > 0 ? 1 : 0;
        check(hits > 20, "speed: repeated text proposes");
    }
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K) {
    SuffixIndex d(1u << 18, 3);
    strata::spec::SessionLookupSource s(d);   // the production path, with kinetic's lag
    s.append(history.data(), history.size());
    Sim sm;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        int match = 0;
        const int n = s.propose(nullptr, 0, 0, K, draft.data(), &match);
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++sm.proposals; sm.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        s.append(&target[i], commit);
        i += commit;
        sm.tokens += (long) commit;
        ++sm.steps;
    }
    return sm;
}

}  // namespace

int main(int argc, char** argv) {
    int K = 8;
    char** files = nullptr;
    int n_files = 0;
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int base = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); base = 4; }
        files = argv + base;
        n_files = argc - base;
    } else {
        unit_tests();
        return g_fail ? 1 : 0;
    }
    std::printf("%-28s %8s %8s %10s %10s              %8s %10s %10s\n", "file", "tokens", "tok/step", "proposed",
                "accepted", "tok/step", "proposed", "accepted");
    for (int f = 0; f < n_files; ++f) {
        const std::vector<int32_t> ids = read_ids(files[f]);
        if (ids.size() < 16) continue;
        const size_t half = ids.size() / 2;
        const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K);
        const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
        const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K);
        const char* name = std::strrchr(files[f], '/') ? std::strrchr(files[f], '/') + 1 : files[f];
        std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                    (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                    c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                    100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
    }
    return 0;
}
