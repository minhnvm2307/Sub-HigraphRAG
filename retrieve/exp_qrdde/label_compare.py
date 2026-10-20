"""Compare the training-label sources on our subgraphs (CPU, no LLM), per question of exp_qrdde/data/<ds>/<split>.pkl:
  SP       SubgraphRAG shortest-path labels (rec['label'], what every retriever here trains on)
  hybrid   REL-RAG rational paths labelled by GPT-4o + DeepSeek (data/<ds>/rational_<split>.pkl; WebQSP train/val only)
  SPARQL   gold-query chain bound to the subgraph (sparql_labels.bind_to_subgraph; WebQSP from InferentialChain +
           Constraints, CWQ from the gold SPARQL), alone ('chain') and with the bound constraint triples ('chain+cons')
Metrics (questions with >= 1 gold answer in the graph; a label set that is empty for a question counts as covering
nothing):
  #triples                 labelled triples per question
  ans>=1 / ans=all         share of questions whose labelled triples touch at least one / every gold answer
  topic->all               every gold answer reachable from a topic entity through labelled triples only
  hops topo / sem          length of the labelled path from a topic to the nearest gold answer (sem: CVT steps not
                           counted); distribution 1 / 2 / 3 / 4+
  overlap                  share of SPARQL chain triples that the other label set also marks; share of the other label
                           set that lies on the SPARQL chain or its constraints
Also writes data/<ds>/sparql_<split>.pkl = {id: {'chain': [triple idx], 'cons': [triple idx]}} for training.
  python label_compare.py <webqsp|cwq> <train|val|test>
"""
import collections, os, pickle, sys
import numpy as np
import sparql_coverage as C
from sparql_labels import bind_to_subgraph

D = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde/data'


def walk(rec, idx, sources):
    """undirected BFS over the labelled triples only -> {entity: (topological hops, semantic hops)}"""
    adj = collections.defaultdict(list)
    for k in idx:
        h, t = int(rec['h'][k]), int(rec['t'][k]); adj[h].append(t); adj[t].append(h)
    d = {s: (0, 0) for s in sources}; dq = collections.deque(sources)
    while dq:
        u = dq.popleft(); tp, sm = d[u]
        for v in adj[u]:
            if v not in d:
                d[v] = (tp + 1, sm + (v < rec['n_text'])); dq.append(v)   # entering a CVT is not a semantic step
    return d


def stats(rec, idx):
    gold = set(rec.get('a_ent', []))
    E = {int(x) for k in idx for x in (rec['h'][k], rec['t'][k])}
    d = walk(rec, idx, list(rec['q_ent']))
    reach = [d[a] for a in gold if a in d]
    return {'n': len(idx), 'any': bool(gold & E), 'all': gold <= E, 'topic_all': all(a in d for a in gold),
            'topo': min(r[0] for r in reach) if reach else None, 'sem': min(r[1] for r in reach) if reach else None}


def main():
    ds, split = sys.argv[1], sys.argv[2]
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/{split}.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else C.cwq_labels(split, recs, m2n)[0]
    rat_p = f'{D}/{ds}/rational_{split}.pkl'
    rational = pickle.load(open(rat_p, 'rb')) if os.path.exists(rat_p) else None
    S = collections.defaultdict(lambda: collections.defaultdict(list)); save = {}
    n_q = 0
    for qid, rec in recs.items():
        if not rec.get('a_ent'):
            continue
        n_q += 1
        N, L = rec['entity_names'], rec['relation_list']
        sets = {'SP': set(np.nonzero(rec['label'])[0].tolist())}
        if rational is not None:
            sets['hybrid'] = set(rational.get(qid, []))
        if qid in labels:
            trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
            ix = {tr: k for k, tr in enumerate(trip)}
            lab = labels[qid]
            lab = dict(lab, constraints=[dict(c, object='?v' if c['object'] == '?unmapped' else c['object']) for c in lab['constraints']])
            b = bind_to_subgraph(lab, trip)
            ch = {ix[t] for t in b['chain_triples'] if t in ix}; co = {ix[t] for t in C.keep_constraints(lab, b['constraint_triples']) if t in ix}
            save[qid] = {'chain': sorted(ch), 'cons': sorted(co)}
        else:
            ch, co = set(), set()
        sets['SPARQL chain'] = ch; sets['SPARQL chain+cons'] = ch | co
        for name, idx in sets.items():
            for k, v in stats(rec, idx).items():
                S[name][k].append(v)
        for name in [x for x in sets if not x.startswith('SPARQL')]:
            if ch:
                S[name]['chain_also'].append(len(ch & sets[name]) / len(ch))
            if sets[name]:
                S[name]['on_query'].append(len(sets[name] & (ch | co)) / len(sets[name]))
    pickle.dump(save, open(f'{D}/{ds}/sparql_{split}.pkl', 'wb'))
    pct = lambda v: f'{100 * np.mean(v):.1f}'
    print(f'### {ds} {split}: {n_q} questions with a gold answer in the graph; SPARQL label bound for '
          f'{sum(1 for i in save if save[i]["chain"])} ({len(labels)} parsed)\n')
    print('| label | #triples mean / median | ans>=1 | ans=all | topic->all answers | hops topo 1/2/3/4+ | hops sem 1/2/3+ | '
          'SPARQL-chain triples also marked | marked triples on the SPARQL query |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, s in S.items():
        topo = [x for x in s['topo'] if x is not None]; sem = [x for x in s['sem'] if x is not None]
        h = lambda v, ks: ' / '.join(f"{100 * np.mean([(x == k) if k != ks[-1] else (x >= k) for x in v]):.0f}" for k in ks)
        print(f"| {name} | {np.mean(s['n']):.1f} / {np.median(s['n']):.0f} | {pct(s['any'])} | {pct(s['all'])} | "
              f"{pct(s['topic_all'])} | {h(topo, [1, 2, 3, 4])} | {h(sem, [1, 2, 3])} | "
              f"{pct(s['chain_also']) if s['chain_also'] else '—'} | {pct(s['on_query']) if s['on_query'] else '—'} |")
    print()


if __name__ == '__main__':
    main()
