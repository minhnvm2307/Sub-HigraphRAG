"""What is actually inside a top-k context, and how much of the budget the score ties away (docs/16 experiment E1).

Motivation. The one context feature that predicts a wrong LLM answer is the NUMBER OF NAMED CANDIDATES
(p = 0.008 on 110 questions, docs/14); the rank of the answer triple does not (p = 0.77). And because pi depends
on (q, r, direction) only, sibling triples sharing a (head, relation) are scored EXACTLY alike, so a single
(head, relation) fan-out can take the whole budget. This script measures that directly, per run, so the headroom
of a budget-aware selector is known before anything is trained.

Per question, over the top-k triples it reports: how many distinct scores there are (tie rate), the largest group
of triples sharing one score, the largest (head, relation) fan-out, how many distinct relations appear, how many
distinct NAMED entities appear (the quantity to minimise), the share of triples touching a topic entity, and
whether an in-graph answer appears at all.

A Freebase entity whose display name is still a machine id (m.xxx / g.xxx) is counted as unnamed: those are the
CVT / mediator nodes, which cannot be an answer. Pass --mid-regex to change the convention.

  python context_anatomy.py --pred ../runs/<run>/pred_cwq_base.pth --pred ../runs/<other>/pred_cwq_base.pth -k 100
"""
import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import torch


def is_named(name, mid):
    return bool(name) and not mid.match(str(name))


def anatomy(entry, k, mid):
    triples = entry['scored_triples'][:k]
    if not triples:
        return None
    scores = np.array([x[3] for x in triples], dtype=np.float64)
    rounded = np.round(scores, 6)                              # ties up to float noise
    tie_groups = Counter(rounded.tolist())
    head_relation = Counter((x[0], x[1]) for x in triples)
    relations = {x[1] for x in triples}
    named = {x[i] for x in triples for i in (0, 2) if is_named(x[i], mid)}
    topics = set(entry.get('q_entity') or [])
    answers = set(entry.get('a_entity_in_graph') or [])
    touching = sum(1 for x in triples if x[0] in topics or x[2] in topics)
    return {
        'n': len(triples),
        'distinct_scores': len(tie_groups),
        'tie_rate': 1.0 - len(tie_groups) / len(triples),
        'largest_tie': max(tie_groups.values()),
        'largest_hr_fanout': max(head_relation.values()),
        'distinct_relations': len(relations),
        'named_candidates': len(named),
        'frac_touching_topic': touching / len(triples),
        'answer_present': float(bool(answers & named)) if answers else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pred', action='append', required=True, help='pred_<ds>_<mode>.pth; repeat to compare runs')
    ap.add_argument('-k', type=int, default=100)
    ap.add_argument('--mid-regex', default=r'^[mg]\.[0-9a-z_]+$')
    ap.add_argument('--csv', default=None, help='also write the per-question rows here')
    a = ap.parse_args()
    mid = re.compile(a.mid_regex)
    fields = ['distinct_scores', 'tie_rate', 'largest_tie', 'largest_hr_fanout', 'distinct_relations',
              'named_candidates', 'frac_touching_topic', 'answer_present']
    rows, summary = [], {}
    for path in a.pred:
        pred = torch.load(path, map_location='cpu', weights_only=False)
        per_question = defaultdict(list)
        for qid, entry in pred.items():
            stats = anatomy(entry, a.k, mid)
            if stats is None:
                continue
            rows.append({'run': path, 'qid': qid, **stats})
            for f in fields:
                per_question[f].append(stats[f])
        summary[path] = {f: float(np.nanmean(per_question[f])) for f in fields}
        summary[path]['questions'] = len(per_question[fields[0]])

    width = max(len(f) for f in fields) + 2
    print(f'top-{a.k}\n')
    print('metric'.ljust(width) + ''.join(f'{i:>14}' for i in range(len(a.pred))))
    for f in ['questions'] + fields:
        print(f.ljust(width) + ''.join(f'{summary[p][f]:>14.3f}' for p in a.pred))
    print()
    for i, p in enumerate(a.pred):
        print(f'  [{i}] {p}')
    print('\nRead it as: fewer named_candidates and larger distinct_relations at the same answer_present is a '
          'better context (docs/14 §19-20). largest_tie and largest_hr_fanout bound what a budget-aware '
          'selector can recover.')
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'per-question rows -> {a.csv}')


if __name__ == '__main__':
    main()
