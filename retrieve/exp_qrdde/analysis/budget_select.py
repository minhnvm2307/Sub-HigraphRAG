"""Re-select the top-k context from an existing pred_*.pth under a budget-aware objective (docs/16 experiment E2).

No retraining and no GPU: it reads the top-500 the retriever already wrote and chooses k of them differently.

Why a different objective at all. The score of a triple is the best complete walk through it, and pi sees only
(q, r, direction), so every sibling of a (head, relation) fan-out carries the SAME score. Ranking by score alone
therefore spends the budget on one fan-out. Selecting a SET under a coverage objective spends it on distinct
pieces of evidence instead, which is what the measured predictor of LLM error asks for (number of named
candidates, docs/14).

Modes:
  topk      baseline: the original order (use it as the control, and to confirm this script is lossless)
  cap       the rule measured in docs/12 §19: at most --cap triples per relation, drop a repeated entity pair,
            refill from the rest. Reported there as +16 of 110 questions over the control, p = 0.06 / 0.44
  cover     greedy maximisation of a monotone submodular coverage objective over groups
                F(S) = sum over groups g of max{ gain(e) : e in S, group(e) = g }
            so greedy is (1 - 1/e)-optimal (verified in analysis/ref_semirings.py, claim C-C). --group picks what
            a group is; --gain picks how a rank becomes a non-negative weight

Grouping (--group): hr = (head, relation) -- the fan-out that ties; rel = relation; pair = unordered entity pair;
hr+pair = both (a triple covers one group of each kind).

The output keeps the format of infer.py, so retrieve/eval.py and reason/prompt_lab.py read it unchanged.

  python budget_select.py --pred ../runs/<run>/pred_cwq_base.pth --mode cover --group hr -k 100 \
      --out ../runs/<run>/pred_cwq_base_cover.pth
  cd ../.. && python eval.py -d cwq -p exp_qrdde/runs/<run>/pred_cwq_base.pth \
      -p exp_qrdde/runs/<run>/pred_cwq_base_cover.pth
"""
import argparse
from collections import defaultdict

import numpy as np
import torch


def gains(scores, kind):
    """rank or score -> strictly positive weight. Reciprocal rank is scale-free, which matters because the RBF
    score is an unnormalised log-probability (every value is negative and the scale moves with the question)."""
    if kind == 'rr':
        return 1.0 / (1.0 + np.arange(len(scores)))
    if kind == 'softmax':
        z = scores - scores.max()
        return np.exp(z) / np.exp(z).sum()
    raise ValueError(kind)


def group_keys(triples, kind):
    """-> list of tuples of group ids; a triple may belong to one group per kind"""
    out = []
    for h, r, t, _ in triples:
        keys = []
        if 'hr' in kind:
            keys.append(('hr', h, r))
        if 'rel' in kind:
            keys.append(('rel', r))
        if 'pair' in kind:
            keys.append(('pair', *sorted((str(h), str(t)))))
        out.append(tuple(keys))
    return out


def select_cover(triples, k, group_kind, gain_kind):
    """greedy coverage. Each step takes the triple with the largest marginal gain; ties fall back to the original
    order, so the result is deterministic and reduces to `topk` when every triple is its own group."""
    scores = np.array([x[3] for x in triples], dtype=np.float64)
    gain = gains(scores, gain_kind)
    keys = group_keys(triples, group_kind)
    covered = defaultdict(float)
    chosen, taken = [], np.zeros(len(triples), dtype=bool)
    for _ in range(min(k, len(triples))):
        best_i, best_delta = -1, -1.0
        for i in range(len(triples)):
            if taken[i]:
                continue
            delta = sum(max(0.0, gain[i] - covered[g]) for g in keys[i])
            if delta > best_delta + 1e-15:
                best_i, best_delta = i, delta
        if best_i < 0:
            break
        taken[best_i] = True
        chosen.append(best_i)
        for g in keys[best_i]:
            covered[g] = max(covered[g], gain[best_i])
    return chosen


def select_cap(triples, k, cap):
    """docs/12 §19 rule: at most `cap` triples per relation, skip an entity pair already used in either direction,
    then refill from what was skipped (relation cap first, pair duplicates last)."""
    per_relation, seen_pairs = defaultdict(int), set()
    chosen, deferred_cap, deferred_pair = [], [], []
    for i, (h, r, t, _) in enumerate(triples):
        pair = tuple(sorted((str(h), str(t))))
        if per_relation[r] >= cap:
            deferred_cap.append(i)
        elif pair in seen_pairs:
            deferred_pair.append(i)
        else:
            chosen.append(i)
            per_relation[r] += 1
            seen_pairs.add(pair)
        if len(chosen) == k:
            return chosen
    for pool in (deferred_cap, deferred_pair):
        for i in pool:
            if len(chosen) == k:
                return chosen
            chosen.append(i)
    return chosen[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='cover', choices=['topk', 'cap', 'cover'])
    ap.add_argument('-k', type=int, default=100)
    ap.add_argument('--group', default='hr', choices=['hr', 'rel', 'pair', 'hr+pair', 'rel+pair'])
    ap.add_argument('--gain', default='rr', choices=['rr', 'softmax'])
    ap.add_argument('--cap', type=int, default=8, help='--mode cap: triples per relation')
    ap.add_argument('--keep', type=int, default=500, help='how many triples to write per question')
    a = ap.parse_args()
    pred = torch.load(a.pred, map_location='cpu', weights_only=False)
    out, moved = {}, []
    for qid, entry in pred.items():
        triples = entry['scored_triples']
        if a.mode == 'topk':
            order = list(range(min(a.k, len(triples))))
        elif a.mode == 'cap':
            order = select_cap(triples, a.k, a.cap)
        else:
            order = select_cover(triples, a.k, a.group, a.gain)
        rest = [i for i in range(len(triples)) if i not in set(order)]
        # the selected k come first, in selection order; the rest keep their original order behind them, so a
        # larger budget downstream still sees a sensible ranking
        picked = order + rest
        out[qid] = {**entry, 'scored_triples': [triples[i] for i in picked[:a.keep]]}
        moved.append(len(set(order) - set(range(min(a.k, len(triples))))))
    torch.save(out, a.out)
    print(f'{len(out)} questions -> {a.out}')
    print(f'mode={a.mode} k={a.k}' + (f' group={a.group} gain={a.gain}' if a.mode == 'cover' else '')
          + (f' cap={a.cap}' if a.mode == 'cap' else ''))
    print(f'triples promoted into the top-{a.k} that were not there before: mean {np.mean(moved):.1f}, '
          f'max {max(moved) if moved else 0}')
    print('Next: retrieve/eval.py on the old and the new file (answer recall must not drop), then '
          'analysis/context_anatomy.py on both (named_candidates must drop).')


if __name__ == '__main__':
    main()
