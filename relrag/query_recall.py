"""Retrieval scored against the GOLD QUERY (SPARQL / WebQSP InferentialChain + Constraints) instead of shortest paths.
On every test question whose gold chain can be walked inside its subgraph (sparql_labels.bind_to_subgraph; see
retrieve/exp_qrdde/sparql_coverage.py for how the labels are built), and that every listed export covers:
  Chain-R@k   recall of the gold-chain triples (all trails from the topic along the gold predicate chain)
  Cons-R@k    recall of the gold-constraint triples (questions with >= 1 bound constraint triple)
  Q-P / Q-F1  precision / F1 of the top-k against chain + constraint triples (per question, averaged)
  PathHit@k   the top-k holds every triple of the gold trail to at least one gold answer (a complete reasoning path)
  PathAll@k   ... to every gold answer reachable along the chain
Triples are matched by (head name, relation, tail name), like GPT-R in gate_report.py: exports that return a
relation in the reverse direction (RAPL / REL-RAG) are under-counted (EXP-36).
  python query_recall.py <webqsp|cwq_transfer|cwq|cwq_to_webqsp> name=path ...   ('REF' = official SubgraphRAG)
"""
import collections, pickle, sys
import numpy as np, torch
sys.path.insert(0, '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde')
import gate_report as G
import sparql_coverage as C

K = [50, 100, 200]
D = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde/data'


def trails(lab, trip):
    """walk the gold chain inside the subgraph -> {end node: [triples of its trail]} (as bind_to_subgraph)"""
    by_sp, by_po = collections.defaultdict(list), collections.defaultdict(list)
    for h, r, t in trip:
        by_sp[(h, r)].append(t); by_po[(r, t)].append(h)
    frontier = {lab['topic']: []}
    for hop in lab['chain']:
        nxt = {}
        for node, tr in frontier.items():
            if hop['direction'] == 'fwd':
                for t in by_sp.get((node, hop['predicate']), []):
                    nxt[t] = tr + [(node, hop['predicate'], t)]
            else:
                for h in by_po.get((hop['predicate'], node), []):
                    nxt[h] = tr + [(h, hop['predicate'], node)]
        frontier = nxt
        if not frontier:
            break
    cons = set()
    nodes = set(frontier) | {lab['topic']} | {x for tr in frontier.values() for t in tr for x in (t[0], t[2])}
    for c in lab['constraints']:
        o = '?v' if c['object'] == '?unmapped' else c['object']
        for n in nodes:
            for t in by_sp.get((n, c['predicate']), []):
                if not C.MID.match(o) and (o.startswith('?') or o == t):
                    cons.add((n, c['predicate'], t))
            for h in by_po.get((c['predicate'], n), []):
                cons.add((h, c['predicate'], n))
    return frontier, cons


def gold_queries(setting):
    """{qid: (chain triples, constraint triples not on the chain, [trail triples to each gold answer])}"""
    ds = 'webqsp' if setting in ('webqsp', 'cwq_to_webqsp') else 'cwq'
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/test.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else C.cwq_labels('test', recs, m2n)[0]
    Q = {}
    for qid, lab in labels.items():
        rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
        gold = {N[i] for i in rec.get('a_ent', [])}
        if not gold:
            continue
        trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
        front, cons = trails(lab, trip)
        gtr = [set(tr) for n, tr in front.items() if n in gold]
        if not gtr:
            continue
        chain = {t for tr in front.values() for t in tr}
        Q[qid] = (chain, cons - chain, gtr)
    return Q


def main():
    setting = sys.argv[1]
    Q = gold_queries(setting)
    ref = torch.load(G.SETTINGS[setting]['ref'], weights_only=False)
    rows, pres = {}, []
    for spec in sys.argv[2:]:
        name, path = spec.split('=', 1)
        d = ref if path == 'REF' else torch.load(path, weights_only=False)
        rows[name] = {i: [tuple(x[:3]) for x in d[i]['scored_triples'][:max(K)]] for i in Q if i in d}
        pres.append(set(rows[name]))
    ids = sorted(set(Q).intersection(*pres))
    print(f'{setting}: {len(ids)} questions with a bound gold chain reaching a gold answer '
          f'({sum(1 for i in ids if Q[i][1])} with constraint triples)\n')
    hdr = ' | '.join(f'{m}@{k}' for m in ('Chain-R', 'Cons-R', 'Q-F1', 'PathHit', 'PathAll') for k in K)
    print(f'| method | {hdr} | Q-P@50 |'); print('|---|' + '---|' * (5 * len(K) + 1))
    for name, r in rows.items():
        out = collections.defaultdict(list)
        for i in ids:
            chain, cons, gtr = Q[i]
            for k in K:
                T = set(r[i][:k]); allq = chain | cons
                out[('Chain-R', k)].append(len(T & chain) / len(chain))
                if cons:
                    out[('Cons-R', k)].append(len(T & cons) / len(cons))
                tp = len(T & allq); p = tp / max(len(T), 1); rc = tp / len(allq)
                out[('Q-F1', k)].append(0 if tp == 0 else 2 * p * rc / (p + rc)); out[('Q-P', k)].append(p)
                out[('PathHit', k)].append(any(g <= T for g in gtr)); out[('PathAll', k)].append(all(g <= T for g in gtr))
        cells = ' | '.join(f"{100 * np.mean(out[(m, k)]):.1f}" for m in ('Chain-R', 'Cons-R', 'Q-F1', 'PathHit', 'PathAll') for k in K)
        print(f"| {name} | {cells} | {100 * np.mean(out[('Q-P', 50)]):.1f} |", flush=True)


if __name__ == '__main__':
    main()
