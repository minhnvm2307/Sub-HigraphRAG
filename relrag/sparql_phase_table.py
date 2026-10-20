"""Unified retrieval benchmark of the gold-query (SPARQL) label phase (10-15 -> 10-19): ONE set of retrievers for every
setting (a cell without an export is '—'), every row of a setting scored on the same questions (answer in the graph
and present in every available export of the setting: the SubgraphRAG subsets, 1,559 / 2,848):
  ansR@50/100/200/500, SP-F1@50, GPT-R@50          as benchmark_all.py
  PathHit@50/100, PathAll@50, Chain-R@50, Cons-R@50, Q-F1@50   against the gold query, as query_recall.py (questions whose
                                                   gold chain reaches a gold answer inside the subgraph)
  python sparql_phase_table.py   -> runs/bench/sparql_phase.json, runs/bench/sparql_phase.md
"""
import collections, json, os
import numpy as np, torch

import benchmark_all as B
from query_recall import gold_queries

R, Q = B.R, B.Q
SETTINGS = {'webqsp': 'WebQSP (ID)', 'cwq_transfer': 'WebQSP → CWQ (OOD)', 'cwq': 'CWQ (ID)', 'cwq_to_webqsp': 'CWQ → WebQSP (OOD)'}
SRC = {'webqsp': 'webqsp', 'cwq_transfer': 'webqsp', 'cwq': 'cwq', 'cwq_to_webqsp': 'cwq'}       # training dataset


def relrag(run, t, tag=''):
    return f'{R}/{run}/retrieval_{"" if t == "webqsp" else "cwq_"}test_path_bf{tag}/retrieval_result_500.pth'



def first(*paths):
    return next((p for p in paths if os.path.exists(p)), paths[0])


def registry(setting):
    """(retriever, labels, export) — the same retriever list for every setting"""
    src, t = SRC[setting], 'webqsp' if setting in ('webqsp', 'cwq_to_webqsp') else 'cwq'
    run = lambda name: f'{Q}/{src}_{name}'
    adr = lambda name: first(f'{run(name)}/adr_{t}_base_mm1_full.pth', f'{run(name)}/adr_{t}_base_mm1.pth',
                             f'{run(name)}/adr_{t}_base_mm_full.pth', f'{run(name)}/adr_{t}_base_mm.pth')
    rr = (lambda d: relrag(d, t)) if src == 'webqsp' else (lambda d: '')
    return [
        ('SubgraphRAG (tự chạy)', 'SP', f'{run("dde")}/adr_{t}_base.pth'),
        ('REL-RAG (tự chạy)', 'LLM', rr('path_codefaithful')),
        ('REL-RAG (tự chạy)', 'SPARQL v2', rr('path_codefaithful_sparqlv2')),
        ('RAPL (checkpoint tác giả)', 'GPT-4o (tác giả)', relrag(f'authors_ckpt/{src}_model', t)),
        ('RBF_nb_mm', 'SP', adr('rbfnb_s42')),
        ('RBF_nb_mm', 'SPARQL v1', adr('rbfnb_sparql_s42')),
        ('RBF_nb_mm', 'SPARQL v2', adr('rbfnb_sparql2_s42')),
        ('RBF + BGE + WL', 'SP', adr('rbfnb_bge_wl_sp_s42')),
        ('RBF + BGE + WL', 'SPARQL v2', adr('rbfnb_bge_wl_sparql2_s42')),
        # WL = RT_combined + weighted loss (qrdde 'dde+rt' + effective_num); RT relation score from GTE or from the
        # Phase 11 BGE-M3 LoRA retrained on SPARQL v2 positives (bge_sparql_data.py)
        ('RT_combined + WL (GTE)', 'SP', first(f'{run("wlq_s42")}/adr_{t}_base_full.pth', f'{run("wlq_s42")}/adr_{t}_base.pth')),
        ('RT_combined + WL (GTE)', 'SPARQL v2', adr('wl_gte_sparql2_s42')),
        ('RT_combined + WL (BGE SPARQL v2)', 'SPARQL v2', adr('wl_bge2_sparql2_s42'))]


def gold_scores(d, ids, Qg):
    out = collections.defaultdict(list)
    for i in ids:
        chain, cons, gtr = Qg[i]
        r = [tuple(x[:3]) for x in d[i]['scored_triples'][:100]]
        for k in (50, 100):
            T = set(r[:k])
            out[f'PathHit@{k}'].append(any(g <= T for g in gtr))
            if k == 50:
                allq = chain | cons
                out['PathAll@50'].append(all(g <= T for g in gtr))
                out['Chain-R@50'].append(len(T & chain) / len(chain))
                if cons:
                    out['Cons-R@50'].append(len(T & cons) / len(cons))
                tp = len(T & allq); p = tp / max(len(T), 1); rc = tp / len(allq)
                out['Q-F1@50'].append(0 if tp == 0 else 2 * p * rc / (p + rc))
    return {m: 100 * float(np.mean(v)) for m, v in out.items()}


def main():
    results = {}
    for setting in SETTINGS:
        ref, ids, A, sp, gpt = B.labels(setting)
        Qg = gold_queries(setting)
        rows, data = registry(setting), {}
        for name, lab, path in rows:
            if path and os.path.exists(path):
                data[(name, lab)] = torch.load(path, weights_only=False)
            else:
                print('missing', setting, name, lab, path or '(not trained)', flush=True)
        common = [i for i in ids if all(i in d for d in data.values())]
        gids = [i for i in common if i in Qg]
        res = {}
        for name, lab, _ in rows:
            key = f'{name} | {lab}'
            if (name, lab) not in data:
                res[key] = None; continue
            d = data[(name, lab)]
            sc, _ = B.score(d, common, A, sp, gpt, None)
            r = {f'ansR@{k}': 100 * float(np.mean(sc[k]['ansR'])) for k in (50, 100, 200, 500)}
            r['SP-F1@50'] = 100 * float(np.nanmean(sc[50]['SP-F1'])); r['GPT-R@50'] = 100 * float(np.nanmean(sc[50]['GPT-R']))
            r.update(gold_scores(d, gids, Qg))
            res[key] = r
            print(setting, key, flush=True)
        results[setting] = {'n': len(common), 'n_gold': len(gids), 'rows': res}
        data.clear()
    os.makedirs(f'{R}/bench', exist_ok=True)
    json.dump(results, open(f'{R}/bench/sparql_phase.json', 'w'), indent=1)
    cols = ['ansR@50', 'ansR@100', 'ansR@200', 'ansR@500', 'PathHit@50', 'PathHit@100', 'PathAll@50', 'Chain-R@50', 'Cons-R@50',
            'Q-F1@50', 'SP-F1@50', 'GPT-R@50']
    lines = []
    for setting, R_ in results.items():
        lines += [f'### {SETTINGS[setting]} ({R_["n"]} câu; metric theo truy vấn gold trên {R_["n_gold"]} câu)', '',
                  '| Retriever | Nhãn | ' + ' | '.join(cols) + ' |', '|---|---|' + '---|' * len(cols)]
        for key, r in R_['rows'].items():
            lines.append(f'| {key} | ' + ' | '.join(f'{r[c]:.1f}' if r and c in r else '—' for c in cols) + ' |')
        lines.append('')
    open(f'{R}/bench/sparql_phase.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
