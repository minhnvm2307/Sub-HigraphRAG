"""Retrieval gate (docs/03_EVALUATION_PROTOCOL.md §0). Decides whether a candidate retriever is a contribution, and
whether a downstream (LLM) run is worth its cost.

Rule (relaxed 2026-10-05):
  - in-distribution setting (e.g. webqsp): the candidate must NOT be significantly worse than either baseline on
    answer recall at any budget (approximately equal is fine);
  - generalisation setting (WebQSP->CWQ, CWQ->WebQSP, another dataset): the candidate must beat BOTH baselines
    (SubgraphRAG first, REL-RAG) on answer recall at every budget 50/100/200 (paired bootstrap 95% lower bound > 0).
  SP / GPT / GPTp recall, precision and F1 are reported as supporting evidence when they help to judge the method
  (e.g. whether more answers come with more evidence); they are not required to pass.
  Downstream: only when the retrieval gain is large enough to plausibly move Hit/F1. The budget-efficiency table
  (candidate@k vs baseline@2k) is the main hint; equal in-distribution retrieval means no in-distribution run.

Label families: ans (answer recall / hit), SP (forward shortest-path triples), GPT (GPT-4o evidence, exact (h,r,t)),
GPTp (the same evidence as unordered entity pairs). Text match; same questions for every retriever.

Baselines per setting:
  webqsp       SubgraphRAG official checkpoint; REL-RAG P code-faithful (best-first; @200 = prefix of the @500 run,
               verified identical to the separate @50/@100 runs)
  cwq_transfer (all retrievers trained on WebQSP) SubgraphRAG = our MLP+DDE repro trained on WebQSP (no official
               WebQSP-trained CWQ export exists; the repro matches the official ckpt on WebQSP, -0.31 R@50, CI has 0);
               REL-RAG P code-faithful. The official CWQ-trained SubgraphRAG is printed as an in-distribution ceiling.

  python gate_report.py --setting webqsp --candidate aer > runs/gate/gate_webqsp_aer.md
"""
import argparse, pickle
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
RR = f'{ROOT}//relrag/runs/path_codefaithful'
QD = f'{ROOT}/retrieve/exp_qrdde/runs'
RR_A = f'{ROOT}/relrag/runs/authors_ckpt/cwq_model'   # REL-RAG trained on CWQ = the authors' checkpoint
SETTINGS = {
    'webqsp': dict(kind='in_distribution',
        ref=f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
        gpt=f'{ROOT}/retrieve/data_files/webqsp/gpt_triples.pth',
        pkl=f'{ROOT}//relrag/data/test.pkl',
        baselines={'SubgraphRAG (official)': 'REF',
                   'REL-RAG P (code-faithful)': {50: f'{RR}/retrieval_test_path_bf/retrieval_result_50.pth',
                                                 100: f'{RR}/retrieval_test_path_bf/retrieval_result_100.pth',
                                                 None: f'{RR}/retrieval_test_path_bf/retrieval_result_500.pth'}},
        info={'DDE (our SubgraphRAG repro)': f'{QD}/webqsp_dde/retrieval_test.pth'},
        candidates={'aer': ('AER', f'{QD}/webqsp_dde_aer/retrieval_test.pth'),
                    'qrdde': ('QR-DDE', f'{QD}/webqsp_qrdde/retrieval_test.pth'),
                    'relrag_reanchor': ('REL-RAG P + re-anchor', {
                        50: f'{RR}/retrieval_test_path_bf_reanchor/retrieval_result_50.pth',
                        100: f'{RR}/retrieval_test_path_bf_reanchor/retrieval_result_100.pth',
                        None: f'{RR}/retrieval_test_path_bf_reanchor/retrieval_result_500.pth'})}),
    'cwq_transfer': dict(kind='generalisation',
        ref=f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth',
        gpt=f'{ROOT}/reason/gpt_labeled/gpt_labeled_cwq_cleaned.pt',
        pkl=f'{ROOT}//relrag/data/cwq/test.pkl',
        baselines={'SubgraphRAG (repro, trained on WebQSP)': f'{QD}/webqsp_dde/retrieval_cwq_test.pth',
                   'REL-RAG P (code-faithful)': {50: f'{RR}/retrieval_cwq_test_path_bf/retrieval_result_50.pth',
                                                 100: f'{RR}/retrieval_cwq_test_path_bf/retrieval_result_100.pth',
                                                 None: f'{RR}/retrieval_cwq_test_path_bf/retrieval_result_500.pth'}},
        info={'SubgraphRAG official, trained on CWQ (ceiling)': 'REF'},
        candidates={'aer': ('AER', f'{QD}/webqsp_dde_aer/retrieval_cwq_test.pth'),
                    'qrdde': ('QR-DDE', f'{QD}/webqsp_qrdde/retrieval_cwq_test.pth'),
                    'relrag_reanchor': ('REL-RAG P + re-anchor', {
                        50: f'{RR}/retrieval_cwq_test_path_bf_reanchor/retrieval_result_50.pth',
                        100: f'{RR}/retrieval_cwq_test_path_bf_reanchor/retrieval_result_100.pth',
                        None: f'{RR}/retrieval_cwq_test_path_bf_reanchor/retrieval_result_500.pth'})}),
    # trained on CWQ. No CWQ-trained REL-RAG run exists, so SubgraphRAG is the only baseline here.
    'cwq': dict(kind='in_distribution',
        ref=f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth',
        gpt=f'{ROOT}/reason/gpt_labeled/gpt_labeled_cwq_cleaned.pt',
        pkl=f'{ROOT}//relrag/data/cwq/test.pkl',
        baselines={'SubgraphRAG (official)': 'REF',
                   # 10-11: the authors' CWQ checkpoint (HF TianJun1/RAPL cwq.pt) in our REL-RAG repro, path_bf
                   'REL-RAG (authors CWQ ckpt)': {50: f'{RR_A}/retrieval_cwq_test_path_bf/retrieval_result_50.pth',
                                                  100: f'{RR_A}/retrieval_cwq_test_path_bf/retrieval_result_100.pth',
                                                  None: f'{RR_A}/retrieval_cwq_test_path_bf/retrieval_result_500.pth'}},
        info={'DDE (our SubgraphRAG repro, CWQ)': f'{QD}/cwq_dde/adr_cwq_base.pth'},
        candidates={}),
    'cwq_to_webqsp': dict(kind='generalisation',
        ref=f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
        gpt=f'{ROOT}/retrieve/data_files/webqsp/gpt_triples.pth',
        pkl=f'{ROOT}//relrag/data/test.pkl',
        baselines={'SubgraphRAG (repro, trained on CWQ)': f'{QD}/cwq_dde/adr_webqsp_base.pth',
                   'REL-RAG (authors CWQ ckpt)': {50: f'{RR_A}/retrieval_test_path_bf/retrieval_result_50.pth',
                                                  100: f'{RR_A}/retrieval_test_path_bf/retrieval_result_100.pth',
                                                  None: f'{RR_A}/retrieval_test_path_bf/retrieval_result_500.pth'}},
        info={'SubgraphRAG official, trained on WebQSP (ceiling)': 'REF'},
        candidates={}),
}
PRIMARY = ['ansR']
RECALL = ['ansR', 'ansHit', 'SP-R', 'GPT-R', 'GPTp-R']
F1 = ['SP-F1', 'GPT-F1', 'GPTp-F1']
METRICS = ['ansR', 'ansHit', 'SP-R', 'SP-P', 'SP-F1', 'GPT-R', 'GPT-P', 'GPT-F1', 'GPTp-R', 'GPTp-P', 'GPTp-F1']


def prf(ret, lab):
    hit = len(ret & lab)
    r, p = hit / len(lab), (hit / len(ret) if ret else 0.0)
    return r, p, (2 * p * r / (p + r) if hit else 0.0)


def load(spec, ref):
    """spec: 'REF' | path | {budget or None: path}. Returns k -> {id: [triples]} lookup function."""
    if spec == 'REF':
        return lambda k: ref
    if isinstance(spec, str):
        d = torch.load(spec, weights_only=False)
        return lambda k: d
    files = {b: torch.load(p, weights_only=False) for b, p in spec.items()}
    return lambda k: files.get(k, files[None])      # budgets without their own run: prefix of the largest run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='webqsp', choices=list(SETTINGS))
    ap.add_argument('--candidate', default='aer')
    ap.add_argument('--candidate_file', default=None, help='score this export instead (name = --candidate)')
    ap.add_argument('--budgets', default=None, help='default: 50,100,200 (+500 for cwq_transfer, reported only)')
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()
    cfg = SETTINGS[args.setting]
    budgets = [int(b) for b in (args.budgets or ('50,100,200,500' if args.setting == 'cwq_transfer' else '50,100,200')).split(',')]
    ref = torch.load(cfg['ref'], weights_only=False)
    cname, cspec = (args.candidate, args.candidate_file) if args.candidate_file else cfg['candidates'][args.candidate]
    runs = {n: load(s, ref) for n, s in cfg['baselines'].items()}
    runs.update({n: load(s, ref) for n, s in cfg['info'].items()})
    runs[cname] = load(cspec, ref)
    bnames = list(cfg['baselines'])

    ids = sorted(i for i in ref if ref[i]['a_entity_in_graph'] and all(i in runs[n](b) for n in runs for b in budgets))
    n_ref = sum(1 for i in ref if ref[i]['a_entity_in_graph'])
    for n in runs:   # an export from the wrong test set shares no ids with this setting's reference
        cov = sum(1 for i in ref if ref[i]['a_entity_in_graph'] and i in runs[n](budgets[0]))
        if cov < 0.9 * n_ref:
            raise SystemExit(f"'{n}' covers {cov}/{n_ref} questions of setting '{args.setting}' (reference {cfg['ref']}). "
                             f"Is it an export of the right test set? (e.g. cwq_transfer needs adr_cwq_*.pth, webqsp needs adr_webqsp_*.pth)")
    recs = {r['id']: r for r in pickle.load(open(cfg['pkl'], 'rb'))}
    def text(rec, ks):
        N, L = rec['entity_names'], rec['relation_list']
        return {(N[rec['h'][k]], L[rec['r'][k]], N[rec['t'][k]]) for k in ks}
    sp = {i: text(recs[i], {k for p in recs[i]['sp_paths'] for k in p}) for i in ids if i in recs}
    sp = {i: v for i, v in sp.items() if v}
    g = torch.load(cfg['gpt'], weights_only=False)
    gpt = {i: set(map(tuple, (t[:3] for t in g[i]))) for i in ids if g.get(i)}
    gptp = {i: {frozenset((x[0], x[2])) for x in v} for i, v in gpt.items()}
    A = {i: set(ref[i]['a_entity_in_graph']) for i in ids}

    def score(sets):
        out = {}
        ents = {i: {x[0] for x in sets[i]} | {x[2] for x in sets[i]} for i in ids}
        out['ansR'] = np.array([len(A[i] & ents[i]) / len(A[i]) for i in ids])
        out['ansHit'] = np.array([float(bool(A[i] & ents[i])) for i in ids])
        for key, lab, pair in [('SP', sp, False), ('GPT', gpt, False), ('GPTp', gptp, True)]:
            m = np.array([prf({frozenset((x[0], x[2])) for x in sets[i]} if pair else sets[i], lab[i]) for i in lab])
            out[f'{key}-R'], out[f'{key}-P'], out[f'{key}-F1'] = m[:, 0], m[:, 1], m[:, 2]
        out['#trip'] = np.array([len(sets[i]) for i in ids])
        return out

    S = {}   # (method, k) -> scores
    for n in runs:
        for k in budgets:
            d = runs[n](k)
            S[(n, k)] = score({i: {tuple(x[:3]) for x in d[i]['scored_triples'][:k]} for i in ids})

    rng = np.random.default_rng(0)
    def boot(a, b):
        d = a - b; idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5]) * 100
        return d.mean() * 100, lo, hi

    print(f'### Gate: {cname} — setting `{args.setting}` — {len(ids)} questions\n')
    print(f'Labels: SP on {len(sp)} questions, GPT on {len(gpt)}. Budgets: {budgets}. '
          f'Baselines: {", ".join(bnames)}. Reference only: {", ".join(cfg["info"])}.\n')
    print('#### 1. Values (%)\n')
    print('| Method | k | #triples | ' + ' | '.join(METRICS) + ' |')
    print('|---|---|---|' + '---|' * len(METRICS))
    for n in [cname] + bnames + list(cfg['info']):
        for k in budgets:
            s = S[(n, k)]
            print(f'| {n} | {k} | {s["#trip"].mean():.1f} | ' + ' | '.join(f'{s[m].mean() * 100:.2f}' for m in METRICS) + ' |')

    print(f'\n#### 2. {cname} minus baseline at the same k (mean [95% CI], points; ✅ = CI > 0, ❌ = CI < 0, ~ = CI has 0). '
          f'Gate metric: {", ".join(PRIMARY)}; the others are supporting evidence\n')
    cols = RECALL + F1
    print('| vs | k | ' + ' | '.join(cols) + ' |')
    print('|---|---|' + '---|' * len(cols))
    gate_k = [k for k in budgets if k <= 200]
    verdict = {b: {'win': 0, 'tie': 0, 'loss': 0} for b in bnames}
    support = {b: {'win': 0, 'tie': 0, 'loss': 0} for b in bnames}
    for b in bnames:
        for k in budgets:
            cells = []
            for m in cols:
                mu, lo, hi = boot(S[(cname, k)][m], S[(b, k)][m])
                res = 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
                if m in PRIMARY and k in gate_k:
                    verdict[b][res] += 1
                elif m in RECALL:
                    support[b][res] += 1
                cells.append(f'{dict(win="✅", loss="❌", tie="~")[res]} {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]')
            print(f'| {b} | {k} | ' + ' | '.join(cells) + ' |')

    print(f'\n#### 3. Budget efficiency: {cname}@k minus baseline@2k (needs candidate@k >= baseline@2k to expect a downstream gain at k)\n')
    eff = ['ansR', 'SP-R', 'GPT-R', 'GPTp-R']
    print('| vs | k vs 2k | ' + ' | '.join(eff) + ' |')
    print('|---|---|' + '---|' * len(eff))
    for b in bnames:
        for k in budgets:
            if 2 * k not in budgets:
                continue
            cells = []
            for m in eff:
                mu, lo, hi = boot(S[(cname, k)][m], S[(b, 2 * k)][m])
                cells.append(f'{"✅" if lo > 0 else ("❌" if hi < 0 else "~")} {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]')
            print(f'| {b} | {k} vs {2 * k} | ' + ' | '.join(cells) + ' |')

    n = len(PRIMARY) * len(gate_k)
    print(f'\n#### 4. Verdict ({cfg["kind"]}; gate budgets {gate_k}, gate metric {", ".join(PRIMARY)})\n')
    oks = []
    for b in bnames:
        v, sp_ = verdict[b], support[b]
        ok = v['win'] == n if cfg['kind'] == 'generalisation' else v['loss'] == 0
        oks.append(ok)
        need = 'beat at every budget' if cfg['kind'] == 'generalisation' else 'not significantly worse at any budget'
        print(f'- vs **{b}** ({need}): {"PASS" if ok else "FAIL"} — gate cells {v["win"]} win / {v["tie"]} tie / '
              f'{v["loss"]} loss of {n}; supporting recall cells (other metrics, all budgets): '
              f'{sp_["win"]} win / {sp_["tie"]} tie / {sp_["loss"]} loss')
    print(f'- **{cfg["kind"]} gate: {"PASS" if all(oks) else "FAIL"}**. A contribution needs the in-distribution gate '
          f'AND at least one generalisation gate to pass; downstream is a judgement call based on section 3.\n')


if __name__ == '__main__':
    main()
