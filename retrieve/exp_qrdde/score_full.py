"""Full test export of a run (test + test_extra = every test question): nb + mm for RBF runs, relation prior if the run
uses one -> runs/<run>/adr_<dataset>_base_mm1_full.pth (top-500, same format as adr_probe.py), the file
relrag/sparql_phase_table.py reads.
  python score_full.py --run cwq_wl_bge2_sparql2_s42 --dataset cwq [--device cuda:0]
"""
import argparse, json, pickle
import torch
from tqdm import tqdm
from qrdde import ROOT, build_scorer, resolve, to_device, attach_rel_prior

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'])
ap.add_argument('--device', default='cuda:0')
args = ap.parse_args()
R = f'{ROOT}/exp_qrdde/runs/{args.run}'
a = json.load(open(f'{R}/result.json'))['args']
m = build_scorer(a); m.load_state_dict(torch.load(f'{R}/best.pt', map_location=args.device)); m.to(args.device).eval()
if hasattr(m, 'rbf'):
    m.rbf.nb, m.rbf.mm = True, True
recs, emb = resolve(args.dataset, 'test', True)
extra = pickle.load(open(f'{ROOT}/exp_qrdde/data/{args.dataset}/test_extra.pkl', 'rb'))
if a.get('rel_prior', 'none') != 'none':
    attach_rel_prior(recs, args.dataset, 'test', a['rel_prior'], a['dataset'])
    attach_rel_prior(extra, args.dataset, 'test_extra', a['rel_prior'], a['dataset'])
out = {}
with torch.no_grad():
    for rec in tqdm(list(recs) + extra, desc=f'{args.run} {args.dataset}'):
        sc = m(to_device(rec, emb, args.device)); top = torch.topk(sc, min(500, len(sc))).indices.tolist(); sc = sc.cpu()
        N, L = rec['entity_names'], rec['relation_list']
        out[rec['id']] = {'question': rec['question'], 'scored_triples': [(N[rec['h'][i]], L[rec['r'][i]], N[rec['t'][i]], float(sc[i])) for i in top]}
torch.save(out, f'{R}/adr_{args.dataset}_base_mm1_full.pth'); print(args.run, args.dataset, len(out), 'questions')
