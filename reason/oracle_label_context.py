"""Label-ceiling contexts (10-19): feed the gold-query labels themselves to the reasoner, to bound what any retriever
trained on them can reach. For every test question with a SPARQL label (retrieve/exp_qrdde/data/<ds>/sparql{,v2}_test.pkl:
v1 = every trail along the gold chain + constraint triples; v2 = only trails ending at a gold answer + their
constraints), the context is exactly the labelled triples (chain first, then constraints), as a prompt_lab --src export.
Questions without a label (no gold answer in the subgraph) are left out; full-set numbers reuse a retriever run for them.
  python oracle_label_context.py <webqsp|cwq> <v1|v2>   -> custom_retrieval/oracle_<ds>_<v>.pth
"""
import pickle, sys
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
ds, ver = sys.argv[1], sys.argv[2]
recs = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/test.pkl', 'rb'))}
lab = pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/sparql{"" if ver == "v1" else "v2"}_test.pkl', 'rb'))
out = {}
for qid, l in lab.items():
    if not l['chain'] or qid not in recs:
        continue
    rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
    ks = list(dict.fromkeys(list(l['chain']) + list(l['cons'])))
    n = len(ks)
    out[qid] = {'question': rec['question'],
                'scored_triples': [(N[rec['h'][k]], L[rec['r'][k]], N[rec['t'][k]], float(n - j)) for j, k in enumerate(ks)]}
torch.save(out, f'{ROOT}/reason/custom_retrieval/oracle_{ds}_{ver}.pth')
sizes = sorted(len(v['scored_triples']) for v in out.values())
print(f'{ds} {ver}: {len(out)} questions, triples per question median {sizes[len(sizes) // 2]}, '
      f'mean {sum(sizes) / len(sizes):.1f}, max {sizes[-1]}, > 100 (cut by k=100): {sum(s > 100 for s in sizes)}')
