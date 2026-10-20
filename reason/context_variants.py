"""Context presentation variants for a custom retrieval file (no retraining; the prompt keeps the triple format of the
sys_icl_dc examples). 10-16 diagnosis: on the WebQSP questions where the SubgraphRAG and RBF contexts both hold an
answer, the same fp8 reasoner answers 114/167 discordant questions right with SubgraphRAG's context but 57/167 with
RBF's, so RBF's context misleads the LLM. Variants of the top-k (k = 100):
  pathorder  same triples, re-ordered so each triple is preceded by the triples of its shortest path from a topic
             entity inside the context (chains read contiguously, best-scored first)
  diverse    drop a triple whose (head, tail) pair, in either direction, is already present; at most `cap` triples per
             relation; refilled from the rest of the top-500 export to k triples
  both       diverse, then pathorder
  taumerge   keep the triples within --tau of the best path log-prob (cap k), then present parallel evidence as one
             line: a CVT step pair (x, r1, CVT) + (CVT, r2, y) becomes (x, "r1 > r2", y), and lines sharing the
             endpoint nearer to the topic and the relation are merged into (x, r, [y1, y2, ...]) (SubgraphRAG's
             merge_tuples idea). The LLM then sees one candidate set instead of several equally plausible chains.
  python context_variants.py <top-500 export> <dataset> <variant> <out.pth> [--ids ids.json] [--k 100] [--cap 8]
"""
import argparse, collections, json, math, re
import torch

REF = {'webqsp': 'scored_triples/webqsp_240912_unidir_test.pth', 'cwq': 'scored_triples/cwq_240907_unidir_test.pth'}


def diverse(trips, k, cap):
    out, pairs, per_rel = [], set(), collections.Counter()
    for t in trips:
        p = frozenset((t[0], t[2]))
        if p in pairs or per_rel[t[1]] >= cap:
            continue
        out.append(t); pairs.add(p); per_rel[t[1]] += 1
        if len(out) == k:
            break
    return out


def pathorder(trips, topics):
    adj = collections.defaultdict(list)
    for j, t in enumerate(trips):
        adj[t[0]].append((t[2], j)); adj[t[2]].append((t[0], j))
    prev = {s: None for s in topics if s in adj}; dq = collections.deque(prev)
    while dq:                                                 # BFS tree from the topics over the context's triples
        u = dq.popleft()
        for v, j in adj[u]:
            if v not in prev:
                prev[v] = (u, j); dq.append(v)
    def path_to(node):
        out = []
        while node in prev and prev[node] is not None:
            u, j = prev[node]; out.append(j); node = u
        return out[::-1]
    order, seen = [], set()
    for j, t in enumerate(trips):                             # best-scored first, each with its path from a topic
        near = min((t[0], t[2]), key=lambda n: len(path_to(n)) if n in prev else 99)
        for pj in path_to(near) + [j]:
            if pj not in seen:
                seen.add(pj); order.append(pj)
    return [trips[j] for j in order]


CVT = re.compile(r'^[mg]\.[0-9a-z_]+$')
lp = lambda s: -math.log1p(math.exp(-s)) if s > -30 else s


def taumerge(trips, topics, tau, k):
    if not trips:
        return trips
    b = lp(trips[0][3]); T = [t for t in trips[:k] if lp(t[3]) >= b - tau]
    adj = collections.defaultdict(set)
    for h, r, t, _ in T:
        adj[h].add(t); adj[t].add(h)
    dist = {s: 0 for s in topics if s in adj}; dq = collections.deque(dist)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    d = lambda n: dist.get(n, 99)
    lines, used = [], set()
    by_cvt = collections.defaultdict(list)                     # CVT -> [(neighbour, relation, score, triple)]
    for tr in T:
        h, r, t, s = tr
        if CVT.match(h) and not CVT.match(t):
            by_cvt[h].append((t, r, s, tr))
        if CVT.match(t) and not CVT.match(h):
            by_cvt[t].append((h, r, s, tr))
    for c, nb in by_cvt.items():
        into, outof = {}, {}                                   # one relation per (CVT, neighbour) and direction: a CVT
        for n, r, sc, tr in nb:                                # is often linked to the same entity both ways
            side = outof if tr[0] == c else into
            if n not in side or sc > side[n][1]:
                side[n] = (r, sc)
        best = {n: into.get(n) or outof.get(n) for n in set(into) | set(outof)}
        used.update(tr for _, _, _, tr in nb)                  # every triple of this CVT is absorbed or dropped
        if len(best) < 2:
            continue                                           # dangling CVT: one named neighbour, nothing to join
        near = min(d(n) for n in best)
        for x in [n for n in best if d(n) == near]:
            for y in best:
                if y != x:
                    rx = into.get(x) or outof[x]; ry = outof.get(y) or into[y]     # x -> CVT, then CVT -> y
                    lines.append((x, f'{rx[0]} > {ry[0]}', y, max(rx[1], ry[1])))
    lines += [tr for tr in T if tr not in used and not (CVT.match(tr[0]) or CVT.match(tr[2]))]
    lines += [tr for tr in T if tr not in used and CVT.match(tr[0]) and CVT.match(tr[2])]   # CVT-CVT steps, rare
    groups, order = {}, []
    for h, r, t, s in lines:
        key = ('fwd', h, r) if d(h) <= d(t) else ('rev', r, t)
        val = t if key[0] == 'fwd' else h
        if key not in groups:
            groups[key] = [[], s]; order.append(key)
        if val not in groups[key][0]:
            groups[key][0].append(val)
        groups[key][1] = max(groups[key][1], s)
    out = []
    for key in sorted(order, key=lambda k_: -groups[k_][1]):
        vals, s = groups[key]
        v = vals[0] if len(vals) == 1 else '[' + ', '.join(vals) + ']'
        out.append((key[1], key[2], v, s) if key[0] == 'fwd' else (v, key[1], key[2], s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('export'); ap.add_argument('dataset'); ap.add_argument('variant', choices=['pathorder', 'diverse', 'both', 'taumerge'])
    ap.add_argument('out'); ap.add_argument('--ids'); ap.add_argument('--k', type=int, default=100); ap.add_argument('--cap', type=int, default=8); ap.add_argument('--tau', type=float, default=2.0)
    a = ap.parse_args()
    d = torch.load(a.export, weights_only=False); ref = torch.load(REF[a.dataset], weights_only=False)
    ids = json.load(open(a.ids)) if a.ids else list(ref)
    out = {}
    for i in ids:
        trips = [tuple(t) for t in d[i]['scored_triples']] if i in d else []
        if a.variant == 'taumerge':
            trips = taumerge(trips, ref[i]['q_entity'], a.tau, a.k)
        elif a.variant in ('diverse', 'both'):
            trips = diverse(trips, a.k, a.cap)
        else:
            trips = trips[:a.k]
        if a.variant in ('pathorder', 'both'):
            trips = pathorder(trips, ref[i]['q_entity'])
        out[i] = {'question': ref[i]['question'], 'a_entity': ref[i].get('a_entity', []), 'scored_triples': trips}
    torch.save(out, a.out)
    print(f'{a.variant}: {len(out)} questions, mean {sum(len(v["scored_triples"]) for v in out.values()) / len(out):.1f} triples -> {a.out}')


if __name__ == '__main__':
    main()
