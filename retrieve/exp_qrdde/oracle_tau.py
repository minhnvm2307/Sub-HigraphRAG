"""Ceiling of a terminal constraint factor tau(v|q) for RBF (10-19, report section 7), before any training.
On a question whose gold chain also ends at non-answer siblings, and where one side constraint of the gold query
(an edge at the last hop: the answer or its CVT) holds on a trail to a gold answer but on no trail to a sibling,
a perfect tau pushes down every triple that lies only on sibling trails. Applied to an existing top-500 export:
  demote   sibling-only triples of separable questions -> bottom of the list
  demote+  also lifts the gold constraint triples found in the list to the top
  demote-all  every triple touching a sibling END entity goes to the bottom (a terminal factor that makes the sibling lose
           as a candidate also lowers its other triples), gold constraint triples lifted
Reports, on the val split: separable questions, siblings left in the top-100, ansR@50/@100; writes the two variants as
val context files for prompt_lab (reason/custom_retrieval/dev_<ds>_val_tau{,plus}.pth).
  python oracle_tau.py <webqsp|cwq>
"""
import collections, pickle, sys
import numpy as np, torch
import sparql_coverage as C
from make_sparql_v2 import all_trails

ROOT = '/home/minhnv/Lab/SubgraphRAG'
D = f'{ROOT}/retrieve/exp_qrdde/data'


def satisfied(c, tr, topic, by_sp, by_po):
    nodes = {topic} | {x for t in tr for x in (t[0], t[2])}
    o = '?v' if c['object'] == '?unmapped' else c['object']
    hits = set()
    for n in nodes:
        hits |= {(n, c['predicate'], t) for t in by_sp.get((n, c['predicate']), []) if o.startswith('?') or o == t}
        hits |= {(h, c['predicate'], n) for h in by_po.get((c['predicate'], n), []) if o.startswith('?') or o == h}
    return hits


def main():
    ds = sys.argv[1]
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/val.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else C.cwq_labels('val', recs, m2n)[0]
    dev = torch.load(f'{ROOT}/reason/custom_retrieval/dev_{ds}_val.pth', weights_only=False)
    demote, lift, sibs, st = {}, {}, {}, collections.Counter()
    for qid, lab in labels.items():
        if qid not in dev:
            continue
        rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
        gold = {N[a] for a in rec.get('a_ent', [])}
        trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
        trails, by_sp, by_po = all_trails(lab, trip)
        good = [tr for e, tr in trails if e in gold]
        bad = [(e, tr) for e, tr in trails if e not in gold and not C.MID.match(e) and not e.startswith(('m.', 'g.'))]
        if not good or not bad:
            continue
        st['with_siblings'] += 1
        sep = [c for c in lab['constraints']
               if any(satisfied(c, tr, lab['topic'], by_sp, by_po) for tr in good)
               and not any(satisfied(c, tr, lab['topic'], by_sp, by_po) for _, tr in bad)]
        if not sep:
            continue
        st['separable'] += 1
        keep = {t for tr in good for t in tr}
        demote[qid] = {t for _, tr in bad for t in tr} - keep
        sibs[qid] = {e for e, _ in bad} - gold - {x for t in keep for x in (t[0], t[2])}
        lift[qid] = set().union(*(satisfied(c, tr, lab['topic'], by_sp, by_po) for c in sep for tr in good))
    print(f'{ds} val: {st["with_siblings"]} questions with same-chain siblings, {st["separable"]} separable by one side constraint')

    def rerank(v, qid, plus, all_=False):
        T = [tuple(x) for x in v['scored_triples']]
        if qid not in demote:
            return T
        top = [x for x in T if plus and tuple(x[:3]) in lift[qid]]
        bad = (lambda x: x[0] in sibs[qid] or x[2] in sibs[qid]) if all_ else (lambda x: tuple(x[:3]) in demote[qid])
        bottom = [x for x in T if bad(x) and x not in top]
        mid = [x for x in T if x not in top and x not in bottom]
        return top + mid + bottom

    def measure(out):
        sib, ansr = collections.defaultdict(list), collections.defaultdict(list)
        for qid, v in out.items():
            gold = set(v['a_entity'])
            for k in (50, 100):
                T = v['scored_triples'][:k]; E = {x[0] for x in T} | {x[2] for x in T}
                if gold:
                    ansr[k].append(len(gold & E) / len(gold))
                if qid in demote:
                    sib[k].append(len(sibs[qid] & E))
        return {k: (100 * np.mean(ansr[k]), np.mean(sib[k]) if sib[k] else 0) for k in (50, 100)}

    for name, plus, all_ in [('original', None, False), ('tau', False, False), ('tauplus', True, False), ('tauall', True, True)]:
        out = {i: dict(v, scored_triples=rerank(v, i, plus, all_) if plus is not None else v['scored_triples']) for i, v in dev.items()}
        m = measure(out)
        print(f'  {name:8s} ansR@50 {m[50][0]:.1f}  ansR@100 {m[100][0]:.1f}  | separable questions: sibling END entities in top-50 {m[50][1]:.2f}, top-100 {m[100][1]:.2f}')
        if plus is not None:
            torch.save(out, f'{ROOT}/reason/custom_retrieval/dev_{ds}_val_{name}.pth')


if __name__ == '__main__':
    main()
