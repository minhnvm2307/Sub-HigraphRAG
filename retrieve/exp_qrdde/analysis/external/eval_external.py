"""Zero-shot retrieval on the external multi-hop sets (10-27): GTSQA (Wikidata), GrailQA (Freebase, our 728-question
set), MetaQA-3hop (WikiMovies). Retrievers trained on WebQSP or CWQ, never on these sets.

Reads runs/<run>/pred_<ds>_base.pth (infer.py --top_k 200) and data_ext/<ds>/test.pkl. Every test question counts
(a question missing from a file counts 0). Per question, at k in K:
  ansR    share of the gold answers found as head / tail of the top-k triples (names, as retrieve/eval.py)
  ansHit  at least one gold answer found
  gtR     GTSQA only: recall of the ground-truth answer-subgraph triples (the paper's "GT triple recall")
  gtAll   GTSQA only: every ground-truth triple retrieved
Groups: all questions; n_hops; GTSQA test_type / n_seeds; GrailQA domain / function.
  python analysis/external/eval_external.py      -> analysis/external/external_1027.json + tables on stdout
"""
import json, os, pickle
from collections import defaultdict

import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
RUNS = f'{ROOT}/retrieve/exp_qrdde/runs'
DATASETS = {'gtsqa': 'GTSQA (Wikidata)', 'grailqa_paper': 'GrailQA (Freebase)', 'metaqa3': 'MetaQA-3hop (movies)'}
RETRIEVERS = {   # name -> {trained on: run}
    'SubgraphRAG (DDE)': {'webqsp': 'webqsp_dde', 'cwq': 'cwq_dde'},
    'RBF old best (no phi)': {'webqsp': 'webqsp_rbfnb_bge_wl_sparql2_s42', 'cwq': 'cwq_rbfnb_bge_wl_sparql2_s42'},
    'RBF + phi (best)': {'webqsp': 'abl_bce_stop_scratch_webqsp', 'cwq': 'abl_bce_stop_scratch_cwq'},
    'RBF + phi post-tuned': {'cwq': 'stop_frozen_cwq'},
}
K = [20, 50, 100, 200]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_1027.json')


def groups_of(ds, rec):
    g = [('all', 'all'), ('hops', str(rec['n_hops']))]
    if ds == 'gtsqa':
        g += [('seeds', str(min(rec['n_seeds'], 3)) + ('+' if rec['n_seeds'] >= 3 else '')),
              ('test_type', ' & '.join(rec['test_type']))]
    if ds == 'grailqa_paper':
        g += [('domain', rec['domain']), ('function', 'none' if rec['function'] == 'none' else 'aggregation / comparison')]
    return g


def question_metrics(rec, triples):
    names = rec['names']
    answers = {names[i] for i in rec['a_ent']}
    gold = {tuple(t) for t in rec.get('gold', [])}
    out = {}
    for k in K:
        top = triples[:k]
        ents = {t[0] for t in top} | {t[2] for t in top}
        found = len(answers & ents)
        out[f'ansR@{k}'] = found / len(answers)
        out[f'ansHit@{k}'] = float(found > 0)
        if gold:
            got = {t[:3] for t in top}
            n = sum(g in got or (g[2], g[1], g[0]) in got for g in gold)
            out[f'gtR@{k}'], out[f'gtAll@{k}'] = n / len(gold), float(n == len(gold))
    return out


def main():
    result = {}
    for ds, title in DATASETS.items():
        records = pickle.load(open(f'{ROOT}/data_ext/{ds}/test.pkl', 'rb'))
        result[ds] = {'title': title, 'n': len(records), 'groups': {}, 'scores': {}}
        sizes = defaultdict(int)
        for rec in records:
            for g in groups_of(ds, rec):
                sizes['|'.join(g)] += 1
        result[ds]['groups'] = dict(sizes)
        for name, runs in RETRIEVERS.items():
            for src, run in runs.items():
                path = f'{RUNS}/{run}/pred_{ds}_base.pth'
                if not os.path.exists(path):
                    print(f'missing {path}'); continue
                pred = torch.load(path, weights_only=False)
                acc = defaultdict(lambda: defaultdict(list))
                for rec in records:
                    triples = pred[rec['id']]['scored_triples'] if rec['id'] in pred else []
                    m = question_metrics(rec, triples)
                    for g in groups_of(ds, rec):
                        for key, v in m.items():
                            acc['|'.join(g)][key].append(v)
                result[ds]['scores'][f'{name} | {src}'] = {
                    g: {key: round(100 * float(np.mean(v)), 2) for key, v in ms.items()} for g, ms in acc.items()}
                a = result[ds]['scores'][f'{name} | {src}']['all|all']
                print(f'{ds:14s} {name:24s} {src:7s} ' + '  '.join(f'{k} {a[k]:5.1f}' for k in
                      ['ansR@50', 'ansR@100', 'ansR@200', 'ansHit@200'] + (['gtR@200', 'gtAll@200'] if ds == 'gtsqa' else [])))
    json.dump(result, open(OUT, 'w'), indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
