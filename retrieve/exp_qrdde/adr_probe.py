"""Anchor-decomposed retrieval (ADR) probe: no training, inference only.

Multi-topic (intersection / conjunction) questions are an unseen composition for a WebQSP-trained retriever (98.2% of
WebQSP train questions have one topic, 48% of CWQ test questions have >= 2). ADR scores the question once per single
anchor (each pass looks like a single-topic WebQSP question) and fuses the passes with the max of z-scores, the same
monotone fusion as AER. Combined with AER re-anchoring, each anchor also gets its own bridges.

Modes (single-topic questions reduce to the first two):
  base     one pass, all topics as anchors                         (= the run's normal 1-pass scoring)
  aer      all topics, then + bridges of the top-B triples         (= AER, passes=2)
  adr      max z over {all topics} + {each topic alone}
  adr_aer  max z over {all, all + bridges(all)} + {a, a + bridges(a)} for each topic a
  *_and{mean,min}  the same passes plus one soft-intersection pass: per anchor a, fuse its passes (max z) into f_a,
           then AND = mean_a f_a (or min_a f_a), z-normalised and added to the max fusion (CQD-style t-norm)

  python adr_probe.py --dataset cwq --run webqsp_dde_aer --modes base,aer,adr,adr_aer
Exports runs/<run>/adr_<dataset>_<mode>.pth (top-500, same format as retrieval_test.pth).
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from qrdde import Scorer, build_scorer, resolve, to_device, with_anchors, bridges
import json
from tqdm import tqdm

ROOT = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde'
z = lambda v: (v - v.mean()) / (v.std() + 1e-6)


@torch.no_grad()
def fuse(m, rec, emb, dev, md, B=3):
    """fused triple scores of one question for mode md (see module docstring)"""
    cache = {}
    x0 = to_device(rec, emb, dev, need_lex=hasattr(m, 'rt_dde'))   # once per question; passes differ only in anchors
    def S(anchors):
        key = tuple(sorted(anchors))
        if key not in cache:
            t = torch.zeros(rec['n_ent'], dtype=torch.long, device=dev); t[list(key)] = 1
            cache[key] = m({**x0, 'topic': F.one_hot(t, 2).float()})
        return cache[key]
    def chain(anchors):
        """pass on anchors, then pass on anchors + bridges; returns list of logits"""
        s = S(anchors); new = bridges({**rec, 'q_ent': sorted(anchors)}, s, B)
        return [s] + ([S(set(anchors) | new)] if new else [])
    tops = set(rec['q_ent']); single = [{t} for t in sorted(tops)] if len(tops) > 1 else []
    if md == 'base':
        passes = [S(tops)]
    elif md == 'aer':
        passes = chain(tops)
    elif md == 'adr':
        passes = [S(tops)] + [S(x) for x in single]
    elif md == 'adr_aer':
        passes = chain(tops) + [p for x in single for p in chain(x)]
    elif md.startswith(('adr_and', 'adr_aer_and')) and single:
        per = [[S(x)] for x in single] if md.startswith('adr_and') else [chain(x) for x in single]
        f = torch.stack([torch.stack([z(p) for p in ps]).max(0).values for ps in per])
        inter = f.mean(0) if md.endswith('mean') else f.min(0).values
        passes = ([S(tops)] + [S(x) for x in single] if md.startswith('adr_and')
                  else chain(tops) + [p for ps in per for p in ps]) + [inter]
    elif md.startswith('adr_and'):
        passes = [S(tops)]
    elif md.startswith('adr_aer_and'):
        passes = chain(tops)
    else:
        raise ValueError(md)
    return passes[0] if len(passes) == 1 else torch.stack([z(p) for p in passes]).max(0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'])
    ap.add_argument('--run', required=True)
    ap.add_argument('--modes', default='base,aer,adr,adr_aer')
    ap.add_argument('--B', type=int, default=3)
    ap.add_argument('--topk', type=int, default=500)
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    a = json.load(open(f'{ROOT}/runs/{args.run}/result.json'))['args']
    
    model = build_scorer(a).to(dev)
    model.load_state_dict(torch.load(f'{ROOT}/runs/{args.run}/best.pt', map_location=dev))
    model.eval()
    recs, emb = resolve(args.dataset, 'test', True)
    modes = args.modes.split(',')
    out = {md: {} for md in modes}

    with torch.no_grad():
        for rec in tqdm(recs, desc="Inference retrieval"):
            for md in modes:
                sc = fuse(model, rec, emb, dev, md, args.B)
                top = torch.topk(sc, min(args.topk, len(sc))).indices.cpu().numpy(); scn = sc.cpu().numpy()
                N, L = rec['entity_names'], rec['relation_list']
                out[md][rec['id']] = {'question': rec['question'], 'scored_triples': [
                    (N[rec['h'][i]], L[rec['r'][i]], N[rec['t'][i]], float(scn[i])) for i in top]}
    for md in modes:
        torch.save(out[md], f'{ROOT}/runs/{args.run}/adr_{args.dataset}_{md}.pth')


if __name__ == '__main__':
    main()
