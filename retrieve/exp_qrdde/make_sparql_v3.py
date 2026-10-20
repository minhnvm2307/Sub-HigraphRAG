"""Gold-query labels v3 (10-19) = v2 + chain-anchored bridges, for questions v2 leaves to SP (label_gap.py):
  1. each hop of the gold chain may be walked in the opposite direction when the declared one has no triple
     (Freebase keeps many relations as reverse pairs; 6 WebQSP test questions)
  2. when the walk reaches at least one hop but no trail ends at a gold answer (chain broken at hop >= 2, or the full
     chain ends at non-answers), keep the gold trails to the deepest frontier and add every shortest undirected path
     (<= MAX_BRIDGE hops, inside the subgraph) from that frontier to a gold answer
Questions whose chain cannot be built or breaks at hop 1 keep no label (SP in qrdde.apply_sparql_labels), as in v2.
Output data/<ds>/sparqlv3_<split>.pkl = {id: {'chain', 'cons', 'neg', 'kind'}}, kind in {v2, flip, bridge}.
  python make_sparql_v3.py <webqsp|cwq> <train|val|test>
"""
import collections, pickle, sys
import sparql_coverage as C
from make_sparql_v2 import constraints_on

D = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde/data'
MAX_BRIDGE, MAX_TRAILS = 3, 5000


def walk(lab, by_sp, by_po):
    """trails along the gold chain, a hop flipped when its declared direction is empty; -> (trails, depth, flipped)"""
    trails, flipped = [(lab['topic'], [])], False
    for d, hop in enumerate(lab['chain']):
        for direction in ([hop['direction'], 'rev' if hop['direction'] == 'fwd' else 'fwd']):
            nxt = []
            for node, tr in trails:
                if direction == 'fwd':
                    nxt += [(t, tr + [(node, hop['predicate'], t)]) for t in by_sp.get((node, hop['predicate']), [])]
                else:
                    nxt += [(h, tr + [(h, hop['predicate'], node)]) for h in by_po.get((hop['predicate'], node), [])]
            if nxt:
                flipped |= direction != hop['direction']
                break
        if not nxt:
            return trails, d, flipped
        trails = nxt[:MAX_TRAILS]
    return trails, len(lab['chain']), flipped


def bridges(sources, gold, adj):
    """triples of every shortest undirected path (<= MAX_BRIDGE hops) from any source node to the nearest gold nodes,
    and the sources they start from"""
    dist, pred = {s: 0 for s in sources}, collections.defaultdict(list)
    front = list(sources)
    for d in range(1, MAX_BRIDGE + 1):
        nxt = []
        for u in front:
            for v, tr in adj[u]:
                if v not in dist:
                    dist[v] = d; nxt.append(v)
                if dist[v] == d:
                    pred[v].append((u, tr))
        if gold & set(nxt):
            break
        front = nxt
    hit = [g for g in gold if g in dist and dist[g] > 0]
    if not hit:
        return set(), set()
    keep, starts, stack, seen = set(), set(), list(hit), set(hit)
    while stack:
        v = stack.pop()
        if dist[v] == 0:
            starts.add(v); continue
        for u, tr in pred[v]:
            keep.add(tr)
            if u not in seen:
                seen.add(u); stack.append(u)
    return keep, starts


def main():
    ds, split = sys.argv[1], sys.argv[2]
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/{split}.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else C.cwq_labels(split, recs, m2n)[0]
    v2 = pickle.load(open(f'{D}/{ds}/sparqlv2_{split}.pkl', 'rb'))
    out, S = {}, collections.Counter()
    for qid, rec in recs.items():
        if qid in v2 and v2[qid]['chain']:
            out[qid] = dict(v2[qid], kind='v2'); S['v2'] += 1; continue
        lab = labels.get(qid); N, L = rec['entity_names'], rec['relation_list']
        gold = {N[a] for a in rec.get('a_ent', [])}
        if lab is None or not gold:
            continue
        trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
        ix, by_sp, by_po, adj = {}, collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for k, tr in enumerate(trip):
            ix.setdefault(tr, k)
            by_sp[(tr[0], tr[1])].append(tr[2]); by_po[(tr[1], tr[2])].append(tr[0])
            adj[tr[0]].append((tr[2], tr)); adj[tr[2]].append((tr[0], tr))
        trails, depth, flipped = walk(lab, by_sp, by_po)
        good = [tr for end, tr in trails if end in gold] if depth == len(lab['chain']) else []
        if good:
            kind, chain = 'flip', {t for tr in good for t in tr}
        elif depth >= 1:
            frontier = {end for end, _ in trails}
            btr, starts = bridges(frontier, gold, adj)
            if not btr:
                S['no_bridge'] += 1; continue
            kind, chain = 'bridge', {t for end, tr in trails if end in starts for t in tr} | btr
        else:
            S['broken_at_1'] += 1; continue
        nodes = {lab['topic']} | {x for t in chain for x in (t[0], t[2])}
        cons = constraints_on(lab, nodes, by_sp, by_po) - chain
        out[qid] = {'chain': sorted(ix[t] for t in chain), 'cons': sorted(ix[t] for t in cons), 'neg': [], 'kind': kind}
        S[kind] += 1; S[f'{kind}_triples'] += len(chain)
    pickle.dump(out, open(f'{D}/{ds}/sparqlv3_{split}.pkl', 'wb'))
    n_gold = sum(1 for r in recs.values() if r.get('a_ent'))
    print(f'{ds} {split}: {len(out)} / {n_gold} records with a gold answer labelled (v2 {S["v2"]}, +flip {S["flip"]}, +bridge {S["bridge"]}); '
          f'still SP: broken at hop 1 {S["broken_at_1"]}, no bridge <= {MAX_BRIDGE} {S["no_bridge"]}, no chain {n_gold - len(out) - S["broken_at_1"] - S["no_bridge"]}; '
          f'triples/question bridge {S["bridge_triples"] / max(1, S["bridge"]):.1f}')


if __name__ == '__main__':
    main()
