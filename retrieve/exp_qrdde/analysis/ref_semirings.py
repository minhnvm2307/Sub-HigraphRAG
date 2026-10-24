"""Reference (numpy) implementations and brute-force tests for the propagation variants proposed in docs/16.

`rbf.py` is the torch implementation; this file is the executable SPEC it must agree with. Everything here is
checked against brute-force enumeration of walks on small random graphs, the same way the non-backtracking
max-product was checked when it was added (docs/11, 10-11).

Three claims are tested:

  C-A  max-product over non-backtracking walks WITH a direction-bigram factor psi(d_prev, d_cur) is computed
       exactly by a compressed state of 8 scalars per node: for EACH last-step direction d, the best score and
       the best score with a different last-step tag (Viterbi top-2 per direction).
       Why it matters: psi carries composition-ORDER information with 4 parameters shared by every relation, so
       no parameter is indexed by a relation (or a relation pair) and Proposition 1 of docs/15 still holds.

  C-A2 the same thing in the ONE-state accumulating form that rbf.py implements (constant `start` row plus a
       single per-direction top-2 state, T synchronous rounds).

  C-B  sum-product (logsumexp semiring) over the SAME non-backtracking walk set is computed exactly with mass
       grouped by (node, relation, direction). That gives the exact partition function Z over complete walks,
       hence a normalised conditional likelihood of the gold walk -- the CRF objective of docs/16 section 3.
       Memory is O(number of distinct (node, relation, direction)) <= 2|E|, not O(|V| * |R|).

  C-C  the budget selector of docs/16 section 4 is a monotone submodular coverage objective, so greedy is
       (1 - 1/e)-optimal; checked against exhaustive search on small instances.

Run:  python ref_semirings.py            (fast, ~10s)
      python ref_semirings.py --graphs 800 --seed 1
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

NEG = -1.0e9


# ============================================================ graph generation
def random_graph(rng, n_ent=7, n_rel=3, n_edge=11, n_text=4):
    """small random multi-relational graph: h, r, t arrays + how many entities carry text (ids 0..n_text-1)"""
    h = rng.integers(0, n_ent, n_edge)
    t = rng.integers(0, n_ent, n_edge)
    r = rng.integers(0, n_rel, n_edge)
    keep = h != t                                     # self-loops make "non-backtracking" ill-defined
    return h[keep], r[keep], t[keep], n_ent, n_rel, min(n_text, n_ent)


def random_factors(rng, n_rel):
    """log pi[r, d] <= 0 (one-hop factor), psi[d_prev, d_cur] (direction bigram), chi[d] (first-step bias)"""
    log_pi = -rng.random((n_rel, 2)) * 2.0
    psi = (rng.random((2, 2)) - 0.5) * 1.2
    chi = (rng.random(2) - 0.5) * 0.6
    return log_pi, psi, chi


# ============================================================ brute force
def enumerate_walks(h, r, t, n_ent, anchor, T):
    """every non-backtracking walk of length 1..T from `anchor`, as a list of (relation, direction) steps plus the
    node it ends at. direction 0 = forward (h -> t), 1 = reverse (t -> h). Non-backtracking = a step may not be
    (r, 1-d) right after (r, d), with r the SAME relation (this is the rule rbf.py implements: the tag is
    relation-level, not edge-level)."""
    out_steps = defaultdict(list)                                     # node -> [(relation, direction, next node)]
    for e in range(len(h)):
        out_steps[int(h[e])].append((int(r[e]), 0, int(t[e])))
        out_steps[int(t[e])].append((int(r[e]), 1, int(h[e])))
    walks = []

    def grow(node, steps):
        if len(steps) == T:
            return
        last = steps[-1] if steps else None
        for rel, direction, nxt in out_steps[node]:
            if last is not None and rel == last[0] and direction == 1 - last[1]:
                continue                                              # backtrack
            new = steps + [(rel, direction)]
            walks.append((new, nxt))
            grow(nxt, new)

    grow(anchor, [])
    return walks


def walk_score(steps, log_pi, psi, chi):
    s = chi[steps[0][1]]
    for i, (rel, direction) in enumerate(steps):
        s += log_pi[rel, direction]
        if i:
            s += psi[steps[i - 1][1], direction]
    return s


def brute_force_nodes(h, r, t, n_ent, anchor, T, log_pi, psi, chi, semiring):
    """max (or logsumexp) over all non-backtracking walks ending at each node; the anchor also keeps the empty walk"""
    acc = [[] for _ in range(n_ent)]
    for steps, end in enumerate_walks(h, r, t, n_ent, anchor, T):
        acc[end].append(walk_score(steps, log_pi, psi, chi))
    acc[anchor].append(0.0)                                           # empty walk
    reduce = (lambda v: max(v)) if semiring == 'max' else (lambda v: float(np.logaddexp.reduce(np.array(v))))
    return np.array([reduce(v) if v else NEG for v in acc])


# ============================================================ C-A: compressed max-product, 8 scalars per node
class TopTwoPerDirection:
    """per node and per last-step direction: (best, its tag, best with another tag, that tag).
    tag = (relation, direction) encoded as 2 * relation + direction; -1 means "no walk"."""

    def __init__(self, n_ent):
        self.best = np.full((n_ent, 2), NEG)
        self.best_tag = np.full((n_ent, 2), -1, dtype=np.int64)
        self.second = np.full((n_ent, 2), NEG)
        self.second_tag = np.full((n_ent, 2), -2, dtype=np.int64)

    def excl(self, node, direction, tag):
        """best score at `node` over walks whose last step has direction `direction` and tag != `tag`"""
        if self.best_tag[node, direction] == tag:
            return self.second[node, direction]
        return self.best[node, direction]

    def offer(self, node, direction, value, tag):
        if value <= NEG:
            return
        if value > self.best[node, direction]:
            if self.best_tag[node, direction] != tag:                 # old best survives as the runner-up
                self.second[node, direction] = self.best[node, direction]
                self.second_tag[node, direction] = self.best_tag[node, direction]
            self.best[node, direction] = value
            self.best_tag[node, direction] = tag
        elif tag != self.best_tag[node, direction] and value > self.second[node, direction]:
            self.second[node, direction] = value
            self.second_tag[node, direction] = tag

    def node_best(self, n_ent, anchor):
        out = self.best.max(1).copy()
        out[anchor] = max(out[anchor], 0.0)                           # empty walk
        return out


def compressed_max_product(h, r, t, n_ent, anchor, T, log_pi, psi, chi):
    """max-product over non-backtracking walks with the direction-bigram factor, 8 scalars per node.

    A step over edge e in direction d arrives from node u and must not follow the tag (r_e, 1 - d). Because psi
    depends on the PREVIOUS DIRECTION only, the candidate is
        max over d_prev of [ excl(u, d_prev, (r_e, 1 - d)) + psi[d_prev, d] ] + log_pi[r_e, d]
    and the excluded tag has direction 1 - d, so it can only touch the d_prev = 1 - d group."""
    state = TopTwoPerDirection(n_ent)
    for e in range(len(h)):                                            # first step: no previous direction
        state.offer(int(t[e]), 0, chi[0] + log_pi[r[e], 0], 2 * int(r[e]) + 0)
        state.offer(int(h[e]), 1, chi[1] + log_pi[r[e], 1], 2 * int(r[e]) + 1)
    # keep only the first steps that actually start at the anchor
    first = TopTwoPerDirection(n_ent)
    for e in range(len(h)):
        if int(h[e]) == anchor:
            first.offer(int(t[e]), 0, chi[0] + log_pi[r[e], 0], 2 * int(r[e]) + 0)
        if int(t[e]) == anchor:
            first.offer(int(h[e]), 1, chi[1] + log_pi[r[e], 1], 2 * int(r[e]) + 1)
    state = first
    reached = [state]
    for _ in range(T - 1):
        nxt = TopTwoPerDirection(n_ent)
        for e in range(len(h)):
            rel = int(r[e])
            for direction, (src, dst) in enumerate(((int(h[e]), int(t[e])), (int(t[e]), int(h[e])))):
                forbidden = 2 * rel + (1 - direction)
                cand = NEG
                for d_prev in (0, 1):
                    prev = state.excl(src, d_prev, forbidden)
                    if prev > NEG:
                        cand = max(cand, prev + psi[d_prev, direction])
                if cand > NEG:
                    nxt.offer(dst, direction, cand + log_pi[rel, direction], 2 * rel + direction)
        state = nxt
        reached.append(state)
    out = np.full(n_ent, NEG)
    for state in reached:
        out = np.maximum(out, state.best.max(1))
    out[anchor] = max(out[anchor], 0.0)
    return out


def compressed_max_product_accum(h, r, t, n_ent, anchor, T, log_pi, psi, chi):
    """The variant rbf.py actually implements: ONE accumulating per-direction top-2 state plus a constant `start`
    row, run for T iterations. A step into direction d over edge e takes, as its predecessor value at the source,
        max( start[src] + chi[d],  max over d_prev of [ excl(src, d_prev, (r_e, 1 - d)) + psi[d_prev, d] ] )
    so first steps and later steps are handled by the same expression. Accumulating means the state at iteration i
    holds the best over walks of length <= i, which is what the node score should be."""
    start = np.full(n_ent, NEG)
    start[anchor] = 0.0
    state = TopTwoPerDirection(n_ent)
    for _ in range(T):
        pending = []
        for e in range(len(h)):
            rel = int(r[e])
            for direction, (src, dst) in enumerate(((int(h[e]), int(t[e])), (int(t[e]), int(h[e])))):
                forbidden = 2 * rel + (1 - direction)
                cand = start[src] + chi[direction] if start[src] > NEG else NEG
                for d_prev in (0, 1):
                    prev = state.excl(src, d_prev, forbidden)
                    if prev > NEG:
                        cand = max(cand, prev + psi[d_prev, direction])
                if cand > NEG:
                    pending.append((dst, direction, cand + log_pi[rel, direction], 2 * rel + direction))
        for dst, direction, value, tag in pending:      # applied after the sweep: one synchronous round
            state.offer(dst, direction, value, tag)
    out = state.best.max(1).copy()
    out[anchor] = max(out[anchor], 0.0)
    return out


# ============================================================ C-B: sum-product with mass per (node, relation, direction)
def logaddexp(a, b):
    return float(np.logaddexp(a, b))


def compressed_sum_product(h, r, t, n_ent, anchor, T, log_pi, psi, chi):
    """logsumexp semiring over the SAME walk set. Exclusion by tag is a subtraction in mass space, so the state is
    the mass grouped by (node, direction, relation) -- at most 2 |E| live entries, plus the per-(node, direction)
    total. Returns the log total mass of walks ending at each node (the anchor also counts the empty walk)."""
    def new_state():
        return {'tot': np.full((n_ent, 2), NEG), 'by_tag': defaultdict(lambda: NEG)}

    def offer(state, node, direction, value, rel):
        if value <= NEG:
            return
        state['tot'][node, direction] = logaddexp(state['tot'][node, direction], value)
        key = (node, direction, rel)
        state['by_tag'][key] = logaddexp(state['by_tag'][key], value)

    def excl(state, node, direction, rel_forbidden, dir_forbidden):
        """log of (total mass at (node, direction)) minus (mass whose last tag is (rel_forbidden, dir_forbidden))"""
        total = state['tot'][node, direction]
        if total <= NEG or direction != dir_forbidden:
            return total
        bad = state['by_tag'].get((node, direction, rel_forbidden), NEG)
        if bad <= NEG:
            return total
        diff = total + np.log1p(-np.exp(min(bad - total, -1e-12)))     # log(exp(total) - exp(bad))
        return float(diff) if np.isfinite(diff) else NEG

    state = new_state()
    for e in range(len(h)):
        if int(h[e]) == anchor:
            offer(state, int(t[e]), 0, chi[0] + log_pi[r[e], 0], int(r[e]))
        if int(t[e]) == anchor:
            offer(state, int(h[e]), 1, chi[1] + log_pi[r[e], 1], int(r[e]))
    totals = [state]
    for _ in range(T - 1):
        nxt = new_state()
        for e in range(len(h)):
            rel = int(r[e])
            for direction, (src, dst) in enumerate(((int(h[e]), int(t[e])), (int(t[e]), int(h[e])))):
                acc = NEG
                for d_prev in (0, 1):
                    prev = excl(state, src, d_prev, rel, 1 - direction)
                    if prev > NEG:
                        acc = logaddexp(acc, prev + psi[d_prev, direction])
                if acc > NEG:
                    offer(nxt, dst, direction, acc + log_pi[rel, direction], rel)
        state = nxt
        totals.append(state)
    out = np.full(n_ent, NEG)
    for state in totals:
        for v in range(n_ent):
            for d in (0, 1):
                out[v] = logaddexp(out[v], state['tot'][v, d])
    out[anchor] = logaddexp(out[anchor], 0.0)
    return out


# ============================================================ C-C: budget selection as submodular coverage
def coverage_value(chosen, groups, gain):
    """sum over covered groups of the best gain in that group: a facility-location / weighted-coverage objective,
    monotone and submodular. `groups[i]` is the group id of candidate i (e.g. its (head, relation) class or the
    walk it belongs to); `gain[i]` is its score mapped to a non-negative weight."""
    best = {}
    for i in chosen:
        g = groups[i]
        best[g] = max(best.get(g, 0.0), gain[i])
    return sum(best.values())


def greedy_select(groups, gain, k):
    chosen, remaining = [], set(range(len(gain)))
    current = 0.0
    for _ in range(min(k, len(gain))):
        best_i, best_delta = None, -1.0
        for i in sorted(remaining):
            delta = coverage_value(chosen + [i], groups, gain) - current
            if delta > best_delta + 1e-12:
                best_i, best_delta = i, delta
        chosen.append(best_i)
        remaining.discard(best_i)
        current += best_delta
    return chosen


def exhaustive_select(groups, gain, k):
    best, best_value = None, -1.0
    for combo in itertools.combinations(range(len(gain)), min(k, len(gain))):
        value = coverage_value(list(combo), groups, gain)
        if value > best_value:
            best, best_value = list(combo), value
    return best, best_value


# ============================================================ tests
def test_max_product(rng, n_graphs, T):
    worst = 0.0
    for _ in range(n_graphs):
        h, r, t, n_ent, n_rel, _ = random_graph(rng)
        if len(h) == 0:
            continue
        log_pi, psi, chi = random_factors(rng, n_rel)
        anchor = int(rng.integers(0, n_ent))
        want = brute_force_nodes(h, r, t, n_ent, anchor, T, log_pi, psi, chi, 'max')
        got = compressed_max_product(h, r, t, n_ent, anchor, T, log_pi, psi, chi)
        mask = want > NEG / 2
        if mask.any():
            worst = max(worst, float(np.abs(want[mask] - got[mask]).max()))
        assert (got[~mask] <= NEG / 2).all(), 'claimed a score for an unreachable node'
    return worst


def test_max_product_accum(rng, n_graphs, T):
    """the accumulating variant must agree with brute force too (it is the one rbf.py mirrors)"""
    worst = 0.0
    for _ in range(n_graphs):
        h, r, t, n_ent, n_rel, _ = random_graph(rng)
        if len(h) == 0:
            continue
        log_pi, psi, chi = random_factors(rng, n_rel)
        anchor = int(rng.integers(0, n_ent))
        want = brute_force_nodes(h, r, t, n_ent, anchor, T, log_pi, psi, chi, 'max')
        got = compressed_max_product_accum(h, r, t, n_ent, anchor, T, log_pi, psi, chi)
        mask = want > NEG / 2
        if mask.any():
            worst = max(worst, float(np.abs(want[mask] - got[mask]).max()))
    return worst


def test_sum_product(rng, n_graphs, T):
    worst = 0.0
    for _ in range(n_graphs):
        h, r, t, n_ent, n_rel, _ = random_graph(rng)
        if len(h) == 0:
            continue
        log_pi, psi, chi = random_factors(rng, n_rel)
        anchor = int(rng.integers(0, n_ent))
        want = brute_force_nodes(h, r, t, n_ent, anchor, T, log_pi, psi, chi, 'sum')
        got = compressed_sum_product(h, r, t, n_ent, anchor, T, log_pi, psi, chi)
        mask = want > NEG / 2
        if mask.any():
            worst = max(worst, float(np.abs(want[mask] - got[mask]).max()))
    return worst


def test_selection(rng, n_instances, k):
    worst_ratio = 1.0
    for _ in range(n_instances):
        n = int(rng.integers(6, 10))
        groups = rng.integers(0, 4, n)
        gain = rng.random(n)
        greedy = coverage_value(greedy_select(groups, gain, k), groups, gain)
        _, best = exhaustive_select(groups, gain, k)
        if best > 0:
            worst_ratio = min(worst_ratio, greedy / best)
    return worst_ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--graphs', type=int, default=300)
    ap.add_argument('--instances', type=int, default=120)
    ap.add_argument('--T', type=int, default=3)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    e_max = test_max_product(rng, a.graphs, a.T)
    print(f'C-A  max-product + non-backtracking + direction bigram, {a.graphs} graphs, T={a.T}: '
          f'max |compressed - brute force| = {e_max:.3e}')
    assert e_max < 1e-9, 'compressed max-product does not match brute force'

    e_acc = test_max_product_accum(rng, a.graphs, a.T)
    print(f'C-A2 the accumulating variant rbf.py mirrors, {a.graphs} graphs, T={a.T}: '
          f'max |accumulating - brute force| = {e_acc:.3e}')
    assert e_acc < 1e-9, 'the accumulating variant does not match brute force'

    e_sum = test_sum_product(rng, a.graphs, a.T)
    print(f'C-B  sum-product (partition function) over the same walk set, {a.graphs} graphs, T={a.T}: '
          f'max |compressed - brute force| = {e_sum:.3e}')
    assert e_sum < 1e-7, 'compressed sum-product does not match brute force'

    ratio = test_selection(rng, a.instances, a.k)
    print(f'C-C  greedy vs exhaustive coverage, {a.instances} instances, k={a.k}: '
          f'worst greedy/optimal = {ratio:.4f}  (submodular bound 1 - 1/e = {1 - 1 / np.e:.4f})')
    assert ratio >= 1 - 1 / np.e - 1e-9, 'greedy fell below the submodular bound: objective is not submodular'
    print('\nall three claims hold on the tested instances.')


if __name__ == '__main__':
    main()
