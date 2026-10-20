"""RBF+nb+mm with the BGE relation prior added at test time (no retraining): pi logit += beta * z(cos), where cos is the
fine-tuned BGE-M3 (Phase 11 recipe, LoRA trained on WebQSP train only; relscore_bge.py) similarity of the question with
each candidate relation, standardised per question. beta = 0.5 was picked on WebQSP val (docs/12 section 7).
Writes runs/<run>/adr_<dataset>_base_mm_bge<beta>.pth (top-500, same format as adr_probe.py).
  python bge_bonus_export.py --run webqsp_rbfnb_s42 --dataset cwq [--beta 0.5] [--split test]
"""
import argparse, json, pickle
import torch
from tqdm import tqdm
from qrdde import build_scorer, resolve, to_device

ROOT = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde'
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--beta', type=float, default=0.5); ap.add_argument('--split', default='test')
args = ap.parse_args()
a = json.load(open(f'{ROOT}/runs/{args.run}/result.json'))['args']
m = build_scorer(a); m.load_state_dict(torch.load(f'{ROOT}/runs/{args.run}/best.pt', map_location='cpu')); m.eval()
m.rbf.nb, m.rbf.mm = True, True
recs, emb = resolve(args.dataset, args.split, True)
cos = pickle.load(open(f'{ROOT}/data/bge_p11/scores_ft/{args.dataset}_{args.split}.pkl', 'rb'))
orig, out = m.rbf.relation_logits, {}
with torch.no_grad():
    for rec in tqdm(recs):
        v = torch.from_numpy(cos[rec['id']]).float()
        v = (v - v.mean()) / (v.std() + 1e-6) if len(v) > 1 else v * 0
        m.rbf.relation_logits = lambda x_, v=v: orig(x_) + args.beta * v[:, None]
        sc = m(to_device(rec, emb, 'cpu')); top = torch.topk(sc, min(500, len(sc))).indices.tolist()
        N, L = rec['entity_names'], rec['relation_list']
        out[rec['id']] = {'question': rec['question'],
                          'scored_triples': [(N[rec['h'][i]], L[rec['r'][i]], N[rec['t'][i]], float(sc[i])) for i in top]}
path = f'{ROOT}/runs/{args.run}/adr_{args.dataset}_base_mm_bge{args.beta:g}.pth'
torch.save(out, path)
print(args.dataset, args.split, len(out), 'questions ->', path)
