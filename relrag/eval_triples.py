"""Triple-level retrieval metrics (recall, precision, F1) against three label sets, text-matched, plus answer
recall, for REL-RAG runs and SubgraphRAG on the same questions.

Label sets (per question, as (head, relation, tail) text):
  SP    forward shortest paths topic -> answer (prepare_data.py `sp_paths`, the paper's ShortestPath labels)
  RAT   rational paths: LLM-chosen candidate instances (`--rational`, a gpt_label.py `nodes` file); optional
  GPT   SubgraphRAG's GPT-4o evidence triples (retrieve/data_files/webqsp/gpt_triples.pth); WebQSP only
  GPTp  the GPT evidence as unordered entity pairs (retrieved triples likewise)
Retrieved set at budget b = the first b `scored_triples`; 'all' = the whole list (for path_free, the triples of
the paths kept until STOP). Precision = |retrieved ∩ label| / |retrieved|.
--match NAME adds rows where every run is cut, per question, to the size of NAME's full list, so precision is
compared at equal size; a paired bootstrap gives the 95% CI of each run minus the reference there.
Answer recall uses SubgraphRAG's `a_entity_in_graph` and the formula of retrieve/eval.py, as eval_retrieval_text.py.

usage: python eval_triples.py name=path.pth [...] --budgets 20,50,100,500,all [--match NAME] [--rational file]
"""
import argparse, json, pickle
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
REF = {'webqsp': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
       'cwq': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth'}
GPT = {'webqsp': f'{ROOT}/retrieve/data_files/webqsp/gpt_triples.pth'}
PKL = {'webqsp': f'{ROOT}//relrag/data/test.pkl', 'cwq': f'{ROOT}//relrag/data/cwq/test.pkl'}


def text_triples(rec, paths):
    names, rels = rec['entity_names'], rec['relation_list']
    return {(names[rec['h'][k]], rels[rec['r'][k]], names[rec['t'][k]]) for p in paths for k in p}


def prf(ret, lab):
    hit = len(ret & lab)
    r = hit / len(lab)
    p = hit / len(ret) if ret else 0.0
    return r, p, (2 * p * r / (p + r) if hit else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--dataset', default='webqsp', choices=['webqsp', 'cwq'])
    ap.add_argument('--budgets', default='20,50,100,500,all')
    ap.add_argument('--ref_name', default='SubgraphRAG')
    ap.add_argument('--match', default=None, help='run whose per-question list size every run is cut to')
    ap.add_argument('--rational', default=None, help='gpt_label.py nodes file (paths.jsonl / paths_minlen.jsonl)')
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()

    ref = torch.load(REF[args.dataset], weights_only=False)
    runs = {args.ref_name: ref}
    for r in args.runs:
        name, path = r.split('=', 1)
        runs[name] = torch.load(path, weights_only=False)
    ids = [i for i in ref if ref[i]['a_entity_in_graph'] and all(i in d for d in runs.values())]

    recs = {r['id']: r for r in pickle.load(open(PKL[args.dataset], 'rb'))}
    labels = {'SP': {i: text_triples(recs[i], recs[i]['sp_paths']) for i in ids if i in recs}}
    if args.rational:
        rat = {d['id']: d['paths'] for d in map(json.loads, open(args.rational))}
        labels['RAT'] = {i: text_triples(recs[i], rat[i]) for i in ids if i in recs and rat.get(i)}
    if args.dataset in GPT:
        g = torch.load(GPT[args.dataset], weights_only=False)
        labels['GPT'] = {i: set(map(tuple, g[i])) for i in ids if g.get(i)}
        # GPTp: the same evidence matched as unordered entity pairs, so a triple counts when the retriever holds the
        # pair under another relation or direction (REL-RAG keeps one triple per (h, t) and only follows edges forward)
        labels['GPTp'] = {i: {frozenset((x[0], x[2])) for x in v} for i, v in labels['GPT'].items()}
    labels = {k: {i: v for i, v in d.items() if v} for k, d in labels.items()}

    def retrieved(name, i, b):
        trip = runs[name][i]['scored_triples']
        return {tuple(x[:3]) for x in (trip if b == 'all' else trip[:b])}

    def size(name, i):
        return len(runs[name][i]['scored_triples'])

    rows = [(name, b, lambda i, name=name, b=b: retrieved(name, i, b))
            for name in runs for b in [x if x == 'all' else int(x) for x in args.budgets.split(',')]]
    if args.match:
        rows += [(name, f'={args.match}', lambda i, name=name: retrieved(name, i, size(args.match, i)))
                 for name in runs]

    print(f'{args.dataset} test: {len(ids)} questions with answer in graph, present in all runs; '
          + ', '.join(f'{k} labels on {len(v)} (mean {np.mean([len(x) for x in v.values()]):.1f} triples)'
                      for k, v in labels.items()))
    hdr = f"{'method':28s} {'K':>10s} {'#trip':>6s} {'ansR':>6s}" + ''.join(
        f' | {k:>4s}-R {k:>4s}-P {k:>4s}-F1' for k in labels)
    print(hdr); print('-' * len(hdr))
    per = {}
    for name, b, get in rows:
        A = {i: set(ref[i]['a_entity_in_graph']) for i in ids}
        sets = {i: get(i) for i in ids}
        ans = np.mean([len(A[i] & ({x[0] for x in sets[i]} | {x[2] for x in sets[i]})) / len(A[i]) for i in ids])
        line = f'{name:28s} {str(b):>10s} {np.mean([len(sets[i]) for i in ids]):6.1f} {ans * 100:6.2f}'
        for k, lab in labels.items():
            conv = (lambda t: {frozenset((x[0], x[2])) for x in t}) if k.endswith('p') else (lambda t: t)
            m = np.array([prf(conv(sets[i]), lab[i]) for i in lab])
            per[(name, str(b), k)] = m
            line += ' | ' + ' '.join(f'{v * 100:7.2f}' for v in m.mean(0)[:2]) + f' {m.mean(0)[2] * 100:8.2f}'
        print(line)
    if args.match:
        print(f'\npaired bootstrap at equal size (= {args.match}): run - {args.ref_name}, mean [95% CI]')
        rng = np.random.default_rng(0)
        for k, lab in labels.items():
            n = len(lab); idx = rng.integers(0, n, size=(args.n_boot, n))
            base = per[(args.ref_name, f'={args.match}', k)]
            for name in runs:
                if name == args.ref_name:
                    continue
                d = per[(name, f'={args.match}', k)] - base
                cis = [np.percentile(d[:, j][idx].mean(1), [2.5, 97.5]) * 100 for j in range(3)]
                print(f'  {k:4s} {name:28s} ' + '  '.join(
                    f'{lbl} {d[:, j].mean() * 100:+6.2f} [{cis[j][0]:+.2f}, {cis[j][1]:+.2f}]'
                    for j, lbl in enumerate(['R', 'P', 'F1'])))


if __name__ == '__main__':
    main()
