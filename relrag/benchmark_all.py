"""Retrieval tables of BENCHMARK.md (10-13): every retriever with an export on disk, on the four settings with labels:
WebQSP (train WebQSP, ID), WebQSP -> CWQ (OOD), CWQ (train CWQ, ID), CWQ -> WebQSP (OOD).

Scoring as gate_report.py / bench_matrix.py, on the SAME questions for every row of a setting (answer entity in the
graph AND present in every export of the setting):
  ansR   answer recall: share of the gold answers (a_entity_in_graph) appearing as head/tail of the retrieved triples
  SP-*   recall / precision / F1 of the retrieved triples against the forward shortest-path triples (REL-RAG records);
         P = matched / retrieved, F1 per question then averaged (questions without SP triples skipped)
  GPT-R  recall of the GPT-4o evidence triples, exact (h, r, t)
at k = 50 / 100 / 200 / 500, plus rows with a dynamic budget (RBF: keep triples whose path log-prob is within tau of the
question's best, cap 100; tau picked on the SOURCE validation split), scored on what they keep (#triples reported).
  python benchmark_all.py   -> runs/bench/bench_all.json, runs/bench/bench_all_tables.md
"""
import json, math, os, pickle, sys
import numpy as np, torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
sys.path.insert(0, f'{ROOT}/relrag')
import gate_report as G

R, Q, S = f'{ROOT}/relrag/runs', f'{ROOT}/retrieve/exp_qrdde/runs', f'{ROOT}/retrieve'
WL, RT, CRT = f'{S}/webqsp_wl_effective_num_Oct05-15:23:05', f'{S}/webqsp_rt_combined_Oct05-13:52:20', f'{S}/cwq_rt_combined_gte_Oct05-19:09:58'
K = [50, 100, 200, 500]
MODES = [('base', ''), ('aer', ' + AER'), ('adr', ' + ADR'), ('adr_aer', ' + AER + ADR')]
OUT = f'{R}/bench'
TAU_WEBQSP = 6.0   # smallest tau whose WebQSP-val ansR beats val top-50 (92.9 vs 92.6, ~59 triples), cap 100


def registry(setting):
    """(label, path, dynamic tau or None)"""
    t = 'webqsp' if setting in ('webqsp', 'cwq_to_webqsp') else 'cwq'
    p = '' if t == 'webqsp' else 'cwq_'
    rows = []
    if setting in ('webqsp', 'cwq_transfer'):
        rows += [('SubgraphRAG (official ckpt)' if t == 'webqsp' else 'SubgraphRAG (DDE repro, train WebQSP)',
                  'REF' if t == 'webqsp' else f'{Q}/webqsp_dde/adr_cwq_base.pth', None),
                 ('RAPL / REL-RAG (author ckpt webqsp.pt, path_bf)', f'{R}/authors_ckpt/webqsp_model/retrieval_{p}test_path_bf/retrieval_result_500.pth', None),
                 ('REL-RAG repro (P, path_bf)', f'{R}/path_codefaithful/retrieval_{p}test_path_bf/retrieval_result_500.pth', None),
                 ('REL-RAG repro (P, path_free, ~43 triples)', f'{R}/path_codefaithful/retrieval_{p}test_path_free/retrieval_result_free.pth', None),
                 ('REL-RAG repro (T, triple)', f'{R}/triple_codefaithful/retrieval_{p}test_triple/retrieval_result_500.pth', None)]
        bb = [(f'{Q}/webqsp_dde', 'DDE (SubgraphRAG repro)'), (f'{Q}/webqsp_qrdde', 'QR-DDE'),
              (f'{Q}/webqsp_dde_aer', 'DDE, AER-trained (reanchor_p 0.5)'),
              (RT, 'RT (relation-topic, no weighted loss)'), (WL, 'WL (RT + weighted loss)'),
              (f'{Q}/webqsp_ase_s42', 'ASE (on DDE)'), (f'{Q}/webqsp_wlq_ase_s42', 'WL + ASE'),
              (f'{Q}/webqsp_wlq_aug_s42', 'WL + aug'), (f'{Q}/webqsp_wlq_ase_aug_s42', 'WL + ASE + aug')]
        rb = f'{Q}/webqsp_rbfnb_s42'
        rows += [('RBF+nb (no mm)', f'{rb}/adr_{t}_base.pth', None),
                 ('RBF+nb+mm', f'{rb}/adr_{t}_base_mm_full.pth', None),
                 (f'RBF+nb+mm, dynamic tau={TAU_WEBQSP:g} cap 100', f'{rb}/adr_{t}_base_mm_full.pth', TAU_WEBQSP),
                 ('RBF+nb+mm + BGE prior (beta 0.5)', f'{rb}/adr_{t}_base_mm_bge0.5.pth', None),
                 ('RBF+nb+mm, seed 0', f'{Q}/webqsp_rbfnb_s0/adr_{t}_base_mm.pth', None),
                 ('RBF+nb+mm + answer loss (rejected)', f'{Q}/webqsp_rbfnb_ans_s42/adr_{t}_base_mm1.pth', None)]
    else:
        rows += [('SubgraphRAG (official ckpt)' if t == 'cwq' else 'SubgraphRAG (DDE repro, train CWQ)',
                  'REF' if t == 'cwq' else f'{Q}/cwq_dde/adr_webqsp_base.pth', None),
                 ('RAPL / REL-RAG (author ckpt cwq.pt, path_bf)', f'{R}/authors_ckpt/cwq_model/retrieval_{p}test_path_bf/retrieval_result_500.pth', None)]
        bb = [(f'{Q}/cwq_dde', 'DDE (SubgraphRAG repro)'), (f'{Q}/cwq_qrdde_resume', 'QR-DDE'),
              (CRT, 'RT (relation-topic, no weighted loss, stopped ep 27)'), (f'{Q}/cwq_ase_full_s42', 'ASE (on DDE)')]
        rows += [('RBF+nb+mm (train CWQ)', f'{Q}/cwq_rbfnb_s42/adr_{t}_base_mm1' + ('_full.pth' if t == 'cwq' else '.pth'), None)]
    for d, lab in bb:
        for m, suf in MODES:
            rows.append((lab + suf, f'{d}/adr_{t}_{m}.pth', None))
    return rows


def labels(setting):
    cfg = G.SETTINGS[setting]
    ref = torch.load(cfg['ref'], weights_only=False)
    recs = {r['id']: r for r in pickle.load(open(cfg['pkl'], 'rb'))}
    ids = [i for i in ref if ref[i]['a_entity_in_graph']]
    def text(rec, ks):
        N, L = rec['entity_names'], rec['relation_list']
        return {(N[rec['h'][k]], L[rec['r'][k]], N[rec['t'][k]]) for k in ks}
    sp = {i: text(recs[i], {k for p in recs[i]['sp_paths'] for k in p}) for i in ids if i in recs}
    g = torch.load(cfg['gpt'], weights_only=False)
    gpt = {i: set(map(tuple, (t[:3] for t in g[i]))) for i in ids if g.get(i)}
    return ref, ids, {i: set(ref[i]['a_entity_in_graph']) for i in ids}, {i: v for i, v in sp.items() if v}, gpt


lp = lambda s: -math.log1p(math.exp(-s)) if s > -30 else s


def cut(t, tau, cap=100):
    if not t:
        return t
    b = lp(t[0][3])
    return [x for x in t[:cap] if lp(x[3]) >= b - tau]


def score(d, ids, A, sp, gpt, tau):
    out, present = {}, np.array([i in d for i in ids])
    for k in ([100] if tau is not None else K):
        a, sr, spp, sf, g, nt = [], [], [], [], [], []
        for i in ids:
            t = d[i]['scored_triples'] if i in d else []
            t = cut(t, tau) if tau is not None else t[:k]
            T = {tuple(x[:3]) for x in t}; E = {x[0] for x in t} | {x[2] for x in t}
            a.append(len(A[i] & E) / len(A[i])); nt.append(len(t))
            if i in sp:
                tp = len(sp[i] & T); r_ = tp / len(sp[i]); p_ = tp / max(len(T), 1)
                sr.append(r_); spp.append(p_); sf.append(0.0 if tp == 0 else 2 * p_ * r_ / (p_ + r_))
            else:
                sr.append(np.nan); spp.append(np.nan); sf.append(np.nan)
            g.append(len(gpt[i] & T) / len(gpt[i]) if i in gpt else np.nan)
        out['dyn' if tau is not None else k] = {'ansR': np.array(a), 'SP-R': np.array(sr), 'SP-P': np.array(spp),
                                                'SP-F1': np.array(sf), 'GPT-R': np.array(g), 'ntrip': np.array(nt, float)}
    return out, present


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for setting in ['webqsp', 'cwq_transfer', 'cwq', 'cwq_to_webqsp']:
        ref, ids, A, sp, gpt = labels(setting)
        sc, pres, cache = {}, {}, {}
        for lab, path, tau in registry(setting):
            if path != 'REF' and not os.path.exists(path):
                print('missing', setting, lab, path, flush=True); continue
            if path not in cache:
                cache.clear(); cache[path] = ref if path == 'REF' else torch.load(path, weights_only=False)
            sc[lab], pres[lab] = score(cache[path], ids, A, sp, gpt, tau)
            print(setting, lab, flush=True)
        cache.clear()
        common = np.all(np.stack(list(pres.values())), 0)
        res = {}
        for lab in sc:
            res[lab] = {'n': int(common.sum())}
            for k, mets in sc[lab].items():
                for m, v in mets.items():
                    res[lab][f'{m}@{k}'] = float(np.nanmean(v[common]) * (1 if m == 'ntrip' else 100))
                res[lab][f'ntrip_median@{k}'] = float(np.median(mets['ntrip'][common]))
        results[setting] = res
    json.dump(results, open(f'{OUT}/bench_all.json', 'w'), indent=1)
    write_md(results)


def write_md(results):
    f = lambda v: '—' if v is None else f'{v:.1f}'
    lines = []
    for setting, res in results.items():
        n = next(iter(res.values()))['n']
        lines += [f'### {setting} ({n} câu)', '',
                  '| Method | Budget | ansR@50 | ansR@100 | ansR@200 | ansR@500 | SP-R@50 | SP-P@50 | SP-F1@50 | SP-R@100 | SP-F1@100 | GPT-R@50 | GPT-R@100 |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for lab, r in res.items():
            if 'ansR@dyn' in r:
                lines.append(f"| {lab} | dyn, mean {r['ntrip@dyn']:.0f} / median {r['ntrip_median@dyn']:.0f} triple | "
                             f"{f(r['ansR@dyn'])} (dyn) | | | | {f(r['SP-R@dyn'])} | {f(r['SP-P@dyn'])} | {f(r['SP-F1@dyn'])} | | | {f(r['GPT-R@dyn'])} | |")
            else:
                g = lambda m, k: f(r.get(f'{m}@{k}'))
                lines.append(f"| {lab} | top-k | {g('ansR', 50)} | {g('ansR', 100)} | {g('ansR', 200)} | {g('ansR', 500)} | "
                             f"{g('SP-R', 50)} | {g('SP-P', 50)} | {g('SP-F1', 50)} | {g('SP-R', 100)} | {g('SP-F1', 100)} | "
                             f"{g('GPT-R', 50)} | {g('GPT-R', 100)} |")
        lines.append('')
    open(f'{OUT}/bench_all_tables.md', 'w').write('\n'.join(lines))
    print(f'-> {OUT}/bench_all.json, {OUT}/bench_all_tables.md')


if __name__ == '__main__':
    main()
