"""Retrieval table of runs' base / aer / adr / adr_aer exports (adr_probe.py) next to the SPARQL-phase registry rows,
on the same question set as sparql_phase_table.py (the common ids of every export in the setting).
  python wl_adr_report.py --train cwq --runs cwq_wl_bge2_sparql2_s42 cwq_wl_gte_sparql2_s42
  --train cwq    -> CWQ (ID) + CWQ -> WebQSP (OOD);   --train webqsp -> WebQSP (ID) + WebQSP -> CWQ (OOD)
"""
import argparse, os
import numpy as np, torch

import benchmark_all as B
import sparql_phase_table as T
from query_recall import gold_queries

ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True, choices=['webqsp', 'cwq']); ap.add_argument('--runs', nargs='+', required=True)
args = ap.parse_args()
settings = [('cwq', 'cwq'), ('cwq_to_webqsp', 'webqsp')] if args.train == 'cwq' else [('webqsp', 'webqsp'), ('cwq_transfer', 'cwq')]
for setting, t in settings:
    ref, ids, A, sp, gpt = B.labels(setting); Qg = gold_queries(setting)
    rows = [(f'{run} | {md}', f'{B.Q}/{run}/adr_{t}_{md}.pth') for run in args.runs for md in ('base', 'aer', 'adr', 'adr_aer')]
    rows += [(f'{n} | {l}', p) for n, l, p in T.registry(setting) if p and os.path.exists(p) and not n.startswith(('REL', 'RT_'))]
    for k, p in rows:
        if not os.path.exists(p):
            print('missing', p)
    data = {k: torch.load(p, weights_only=False) for k, p in rows if os.path.exists(p)}
    common = [i for i in ids if all(i in d for d in data.values())]; gids = [i for i in common if i in Qg]
    print(f'\n### {T.SETTINGS[setting]} ({len(common)} q, gold-query metrics on {len(gids)})\n'
          '| run | mode | ansR@50 | @100 | @200 | @500 | PathHit@50 | Cons-R@50 |\n|---|---|---|---|---|---|---|---|')
    for k, d in data.items():
        sc, _ = B.score(d, common, A, sp, gpt, None); g = T.gold_scores(d, gids, Qg)
        print(f'| {k} | ' + ' | '.join(f'{100 * np.mean(sc[x]["ansR"]):.1f}' for x in (50, 100, 200, 500))
              + f' | {g["PathHit@50"]:.1f} | {g.get("Cons-R@50", float("nan")):.1f} |')
