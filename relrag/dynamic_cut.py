"""Dynamic (per-question) retrieval budget: keep the triples whose score is within tau of the question's best score
(score = a logit; RBF-Ret: logit of the complete-path probability, compared in log-probability; triple classifiers
(SubgraphRAG / WL): their own logit, compared in log-sigmoid too, so every method gets the same rule), capped at
`cap`. Reports, per tau, ansR / mean and median #triples / SP-P / SP-F1, next to fixed top-k, so methods can be
compared at matched mean #triples.
  python dynamic_cut.py webqsp "RBF=<export>" "WL=<export>" ...
"""
import math, os, pickle, sys
import numpy as np
import torch
ROOT = '/home/minhnv/Lab/SubgraphRAG'
REF = {'webqsp': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
       'cwq': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth'}
lp = lambda s: -math.log1p(math.exp(-s)) if s > -30 else s


def main():
    ds, cap = sys.argv[1], int(os.environ.get('CAP', 300))
    taus = [float(t) for t in os.environ.get('TAUS', '2,3,4,5,6').split(',')]; ks = [int(k) for k in os.environ.get('KS', '30,50,100,200,300').split(',')]
    recs = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/test.pkl', 'rb'))}
    ref = torch.load(REF[ds], weights_only=False)
    ids = [i for i in recs if i in ref and ref[i]['a_entity_in_graph']]
    gold = {}
    for i in ids:
        r = recs[i]; N, L = r['entity_names'], r['relation_list']; lab = r['label'].astype(bool)
        gold[i] = {(N[h], L[rr], N[t]) for h, rr, t in zip(r['h'][lab], r['r'][lab], r['t'][lab])}
    print('| method | selection | ansR | mean #triples | median | SP-P | SP-F1 |\n|---|---|---|---|---|---|---|')
    for spec in sys.argv[2:]:
        name, path = spec.split('=', 1)
        d = torch.load(path, weights_only=False) if path != 'REF' else {i: {'scored_triples': v['scored_triples']} for i, v in ref.items()}
        def ev(sel):
            ar, n, P, F = [], [], [], []
            for i in ids:
                t = sel(d[i]['scored_triples']) if i in d else []; A = set(ref[i]['a_entity_in_graph'])
                E = {x[0] for x in t} | {x[2] for x in t}; ar.append(len(A & E) / len(A)); n.append(len(t))
                g = gold[i]
                if g:
                    tp = len({tuple(x[:3]) for x in t} & g); p = tp / max(len(t), 1); rc = tp / len(g)
                    P.append(p); F.append(0 if tp == 0 else 2 * p * rc / (p + rc))
            return f'{100 * np.mean(ar):.1f} | {np.mean(n):.0f} | {np.median(n):.0f} | {100 * np.mean(P):.1f} | {100 * np.mean(F):.1f}'
        for k in ks:
            print(f'| {name} | top-{k} | {ev(lambda t, k=k: t[:k])} |')
        for tau in taus:
            def sel(t, tau=tau):
                if not t: return t
                b = lp(t[0][3]); return [x for x in t[:cap] if lp(x[3]) >= b - tau]
            print(f'| {name} | dynamic tau={tau:g} | {ev(sel)} |', flush=True)


if __name__ == '__main__':
    main()
