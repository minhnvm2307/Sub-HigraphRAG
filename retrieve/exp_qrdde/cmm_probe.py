"""Constraint-aware complete-path max-marginal (rbf.cmm_scores), inference only on an RBF run (nb + mm). Constraints =
text entities named in the question that are not topics (lower-cased, non-alphanumerics -> spaces, whole-word match,
name length >= 4). Exports one top-500 file per setting (eps_a for the other topics, eps_c for the named entities).
  python cmm_probe.py --run webqsp_rbfnb_s42 --dataset cwq
"""
import argparse, json, re, torch
from tqdm import tqdm
from qrdde import build_scorer, resolve, to_device, attach_rel_prior

ROOT = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde'
SETTINGS = {'and0.3': (0.3, None), 'and0.1': (0.1, None), 'men0.3': (1.0, 0.3), 'men0.1': (1.0, 0.1), 'and0.3_men0.3': (0.3, 0.3)}


def named_entities(rec):
    q = ' ' + re.sub(r'[^a-z0-9 ]', ' ', rec['question'].lower()) + ' '; tops = set(rec['q_ent']); out = []
    for e, nm in enumerate(rec['entity_names'][:rec['n_text']]):
        if e in tops:
            continue
        s = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', nm.lower())).strip()
        if len(s) >= 4 and f' {s} ' in q:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--settings', default=','.join(SETTINGS))
    args = ap.parse_args(); sets = args.settings.split(',')
    a = json.load(open(f'{ROOT}/runs/{args.run}/result.json'))['args']
    m = build_scorer(a); m.load_state_dict(torch.load(f'{ROOT}/runs/{args.run}/best.pt', map_location='cpu')); m.eval()
    m.rbf.nb, m.rbf.mm = True, True
    recs, emb = resolve(args.dataset, 'test', True)
    if a.get('rel_prior', 'none') != 'none':
        attach_rel_prior(recs, args.dataset, 'test', a['rel_prior'], a['dataset'])
    out = {s: {} for s in sets}; n_named = []
    with torch.no_grad():
        for rec in tqdm(recs):
            x = to_device(rec, emb, 'cpu'); named = torch.tensor(named_entities(rec), dtype=torch.long); n_named.append(len(named))
            N, L = rec['entity_names'], rec['relation_list']
            for s in sets:
                ea, ec = SETTINGS[s]
                sc = m.rbf.cmm_scores(x, named if ec is not None else torch.tensor([], dtype=torch.long), ea, ec if ec else 1.0)
                top = torch.topk(sc, min(500, len(sc))).indices.tolist()
                out[s][rec['id']] = {'question': rec['question'], 'scored_triples': [
                    (N[rec['h'][i]], L[rec['r'][i]], N[rec['t'][i]], float(sc[i])) for i in top]}
    for s in sets:
        torch.save(out[s], f'{ROOT}/runs/{args.run}/cmm_{args.dataset}_{s}.pth')
    print('named entities per question: mean', sum(n_named) / len(n_named), '| share with >= 1:', sum(1 for v in n_named if v) / len(n_named))


if __name__ == '__main__':
    main()
