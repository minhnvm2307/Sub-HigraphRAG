"""Budget table for the 10-12 gate (user): a new retriever counts as a win when its recall@50 beats the OLD methods'
recall@200/300, and retrieval quality is read as recall first, then precision / F1 of the retrieved triples.

For each export (name=path, {id: {'scored_triples': [(h, r, t, score), ...]}}, 'REF' = SubgraphRAG official from
reason/scored_triples) and each budget k: answer recall (ansR), answer hit, and P / R / F1 of the top-k triples against
the shortest-path triples of SubgraphRAG (the label every retriever here trains on; matched by (h, r, t) names).
Also prints the ceiling of ansR@k (best possible when every answer needs its own triple: min(1, k / #answers)).

  python budget_table.py webqsp "SubgraphRAG=REF" "RBF+nb+mm=<export>" ...
"""
import pickle, sys
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
REF = {'webqsp': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
       'cwq': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth'}
K = [50, 100, 200, 300]


def load(path, ds):
    if path != 'REF':
        return torch.load(path, weights_only=False)
    d = torch.load(REF[ds], weights_only=False)
    return {i: {'scored_triples': v['scored_triples']} for i, v in d.items()}


def main():
    ds = sys.argv[1]
    recs = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/test.pkl', 'rb'))}
    ref = torch.load(REF[ds], weights_only=False)
    ids = [i for i in recs if i in ref and ref[i]['a_entity_in_graph']]
    gold = {}
    for i in ids:
        r = recs[i]; N, L = r['entity_names'], r['relation_list']; lab = r['label'].astype(bool)
        gold[i] = {(N[h], L[rr], N[t]) for h, rr, t in zip(r['h'][lab], r['r'][lab], r['t'][lab])}
    ceil = {k: np.mean([min(1.0, k / len(ref[i]['a_entity_in_graph'])) for i in ids]) for k in K}
    print(f'{ds}: {len(ids)} questions with an answer in the graph; ceiling of ansR@k (one triple per answer): '
          + ' / '.join(f'@{k} {100 * ceil[k]:.1f}' for k in K) + '\n')
    print('| method | k | #triples | ansR | ansHit | SP-P | SP-R | SP-F1 |')
    print('|---|---|---|---|---|---|---|---|')
    for spec in sys.argv[2:]:
        name, path = spec.split('=', 1)
        d = load(path, ds)
        for k in K:
            ar, ah, P, R, F, nt = [], [], [], [], [], []
            for i in ids:
                t = d[i]['scored_triples'][:k] if i in d else []
                A = set(ref[i]['a_entity_in_graph']); E = {x[0] for x in t} | {x[2] for x in t}
                ar.append(len(A & E) / len(A)); ah.append(bool(A & E)); nt.append(len(t))
                g = gold[i]
                if g:
                    tp = len({tuple(x[:3]) for x in t} & g); p = tp / max(len(t), 1); rc = tp / len(g)
                    P.append(p); R.append(rc); F.append(0.0 if tp == 0 else 2 * p * rc / (p + rc))
            print(f'| {name} | {k} | {np.mean(nt):.0f} | {100 * np.mean(ar):.1f} | {100 * np.mean(ah):.1f} | '
                  f'{100 * np.mean(P):.1f} | {100 * np.mean(R):.1f} | {100 * np.mean(F):.1f} |', flush=True)


if __name__ == '__main__':
    main()
