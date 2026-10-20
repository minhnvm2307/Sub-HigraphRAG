"""Why do questions with a gold answer in the subgraph get no SPARQL v2 label, and how far is the answer from where the
gold chain stops? (10-19) For every record with a gold answer in its subgraph and no v2 label:
  no_label       the gold query could not be turned into a chain (WebQSP: no InferentialChain; CWQ: parse / topic MID)
  topic_missing  the chain's topic entity is not in the subgraph
  broken@k       the walk along the gold chain dies at hop k (no triple with that predicate from the frontier)
  wrong_end      the whole chain walks, but every trail ends at a non-answer (v2 drops these; mostly an unresolved FILTER)
and the undirected hop distance, inside the subgraph, from the deepest frontier of the chain walk to the nearest gold
answer (the length of the bridge a "chain + shortest bridge" label would need), next to the distance from the topic.
  python label_gap.py <webqsp|cwq> <train|val|test>
"""
import collections, pickle, sys
import sparql_coverage as C

D = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde/data'


def bfs(adj, src, goal, max_d=6):
    seen, front = set(src), set(src)
    for d in range(max_d + 1):
        if front & goal:
            return d
        nxt = {v for u in front for v in adj[u]} - seen
        if not nxt:
            return None
        seen |= nxt; front = nxt
    return None


def main():
    ds, split = sys.argv[1], sys.argv[2]
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/{split}.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    if ds == 'webqsp':
        labels, why = {i: l for i, l in wq.items() if i in recs}, None
    else:
        labels, why = C.cwq_labels(split, recs, m2n)
    v2 = pickle.load(open(f'{D}/{ds}/sparqlv2_{split}.pkl', 'rb'))
    cat, bridge, topic_d, hops_of = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    n_gold = 0
    for qid, rec in recs.items():
        N = rec['entity_names']; gold = set(rec.get('a_ent', []))
        if not gold:
            continue
        n_gold += 1
        if qid in v2 and v2[qid]['chain']:
            continue
        lab = labels.get(qid)
        if lab is None:
            cat['no_label'] += 1; continue
        name2i = collections.defaultdict(list)
        for k, nm in enumerate(N):
            name2i[nm].append(k)
        if lab['topic'] not in name2i:
            cat['topic_missing'] += 1; continue
        h, r, t = rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist(); L = rec['relation_list']
        adj = collections.defaultdict(set); by_sp, by_po = collections.defaultdict(set), collections.defaultdict(set)
        for a, b, c in zip(h, r, t):
            adj[a].add(c); adj[c].add(a); by_sp[(a, L[b])].add(c); by_po[(L[b], c)].add(a)
        front, depth = set(name2i[lab['topic']]), 0
        for hop in lab['chain']:
            nxt = set()
            for u in front:
                nxt |= by_sp[(u, hop['predicate'])] if hop['direction'] == 'fwd' else by_po[(hop['predicate'], u)]
            if not nxt:
                break
            front, depth = nxt, depth + 1
        c = 'wrong_end' if depth == len(lab['chain']) else f'broken@{depth + 1}'
        cat[c] += 1; hops_of[len(lab['chain'])] += 1
        d = bfs(adj, front, gold); bridge[(c if c == 'wrong_end' else 'broken', d)] += 1
        topic_d[bfs(adj, set(name2i[lab['topic']]), gold)] += 1
    miss = sum(cat.values())
    print(f'{ds} {split}: {n_gold} records with a gold answer in the subgraph; {miss} ({100 * miss / max(1, n_gold):.1f}%) without a v2 label')
    for k, v in sorted(cat.items()):
        print(f'   {k:14s} {v:6d}  ({100 * v / max(1, miss):.1f}%)')
    if why:
        print('   CWQ parse failures (all records):', dict(why.most_common(4)))
    fmt = lambda C_: ', '.join(f'{"none" if k is None else k}: {v}' for k, v in sorted(C_.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0)))
    for kind in ('wrong_end', 'broken'):
        sub = collections.Counter({d: v for (k, d), v in bridge.items() if k == kind})
        if sub:
            print(f'   bridge length from the deepest chain frontier to a gold answer [{kind}]: {fmt(sub)}')
    print(f'   distance topic -> gold answer (same questions): {fmt(topic_d)}')
    print(f'   gold chain length of these questions: {dict(sorted(hops_of.items()))}')


if __name__ == '__main__':
    main()
