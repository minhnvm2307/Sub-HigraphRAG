"""Compare retrievers on the same footing: answer recall at a triple budget, text-matched,
using SubgraphRAG's `a_entity_in_graph` and the formula of retrieve/eval.py.

usage: python eval_retrieval_text.py name=path.pth [name=path.pth ...] --budgets 50,100,500
"""
import sys, argparse
import numpy as np
import torch

REF = '/home/minhnv/Lab/SubgraphRAG/reason/scored_triples/webqsp_240912_unidir_test.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--budgets', default='50,100,500')
    ap.add_argument('--ref', default=REF, help='SubgraphRAG scored triples giving a_entity_in_graph (and the SubgraphRAG row)')
    ap.add_argument('--ref_name', default='SubgraphRAG')
    args = ap.parse_args()
    ref = torch.load(args.ref, weights_only=False)
    budgets = [int(b) for b in args.budgets.split(',')]
    runs = {args.ref_name: ref}
    for r in args.runs:
        name, path = r.split('=', 1)
        runs[name] = torch.load(path, weights_only=False)
    ids = [i for i in ref if ref[i]['a_entity_in_graph'] and all(i in d for d in runs.values())]
    print(f'questions with answer in graph and present in all runs: {len(ids)}')
    print(f"{'method':32s}" + ''.join(f'  R@{b:<5d}' for b in budgets) + '  avg#triples(max budget)')
    for name, d in runs.items():
        row = []
        for b in budgets:
            vals = []
            for i in ids:
                A = set(ref[i]['a_entity_in_graph'])
                trip = d[i]['scored_triples'][:b]
                ents = {x[0] for x in trip} | {x[2] for x in trip}
                vals.append(len(A & ents) / len(A))
            row.append(np.mean(vals) * 100)
        ntrip = np.mean([len(d[i]['scored_triples'][:max(budgets)]) for i in ids])
        print(f'{name:32s}' + ''.join(f'  {v:6.2f}' for v in row) + f'  {ntrip:.1f}')


if __name__ == '__main__':
    main()
