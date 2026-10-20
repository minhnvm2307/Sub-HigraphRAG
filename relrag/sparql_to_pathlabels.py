"""Convert gold-query (SPARQL) labels into relrag's training-label format (dataset.py::load_label_paths,
'paths:<file>.jsonl', one line {'id', 'paths': [[line-graph node idx, ...], ...]}), so train.py can use them in place
of the hybrid GPT-4o+DeepSeek rational-path labels (relrag/data/labels_hybrid_gpt4o_deepseek/).

Why not reuse data/<ds>/sparqlv2_<split>.pkl directly: those (retrieve/exp_qrdde/make_sparql_v2.py) are built against
qrdde's own triple indexing and store an unordered set of triple indices covering every trail ('chain'), not one
ordered sequence per trail. train.py's path_loss (Eq. 8) needs one ordered sequence of line-graph node ids per path,
walked here over relrag's own (h, t)-deduped triples (prepare_data.py::build_sample).

v2-style filter: keep only trails that END AT A GOLD ANSWER (not every trail on the gold relation chain).
  <variant>.jsonl        every such trail (paper-faithful: path_loss samples one at random each step, Eq. 8)
  <variant>_minlen.jsonl only the shortest trail(s) (code-faithful: authors' labels keep one chain instance)
  python sparql_to_pathlabels.py webqsp   [--splits train,val]
"""
import argparse, json, pickle, sys
sys.path.insert(0, '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde')
from make_sparql_v2 import all_trails
from sparql_coverage import webqsp_annotations, cwq_labels

D = '/home/minhnv/Lab/SubgraphRAG/relrag/data'


def good_trails(lab, rec):
    N, L = rec['entity_names'], rec['relation_list']
    trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
    ix = {}
    for k, tr in enumerate(trip):
        ix.setdefault(tr, k)                                  # first occurrence (= pair2node order, (h,t) unique)
    trails, _, _ = all_trails(lab, trip)
    gold = set(rec['a_entity'])
    return [[ix[t] for t in tr] for end, tr in trails if end in gold and all(t in ix for t in tr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['webqsp', 'cwq'])
    ap.add_argument('--splits', default='train,val')
    a = ap.parse_args()
    if a.dataset == 'webqsp':
        labels, _ = webqsp_annotations()
        path = lambda split: f'{D}/{"train" if split == "train" else "val"}.pkl'
    else:
        _, mid2name = webqsp_annotations()
        path = lambda split: f'{D}/cwq/{split}.pkl'
    out_d = f'{D}/labels_sparqlv2'; import os; os.makedirs(out_d, exist_ok=True)
    for split in a.splits.split(','):
        recs = {r['id']: r for r in pickle.load(open(path(split), 'rb'))}
        if a.dataset == 'cwq':
            for r in recs.values():
                r['q_ent'] = r['q_entity_id_list']             # cwq_labels' only non-shared key name
            labels, _ = cwq_labels(split, recs, mid2name)
        n_ok = n_empty = n_total = 0
        full, minlen = [], []
        for qid, rec in recs.items():
            lab = labels.get(qid)
            if lab is None:
                continue
            n_total += 1
            trails = good_trails(lab, rec)
            if not trails:
                n_empty += 1; continue
            n_ok += 1
            full.append({'id': qid, 'paths': trails})
            m = min(len(t) for t in trails)
            minlen.append({'id': qid, 'paths': [t for t in trails if len(t) == m][:1]})
        with open(f'{out_d}/{split}.jsonl', 'w') as f:
            f.writelines(json.dumps(d) + '\n' for d in full)
        with open(f'{out_d}/{split}_minlen.jsonl', 'w') as f:
            f.writelines(json.dumps(d) + '\n' for d in minlen)
        print(f'{a.dataset} {split}: {n_total} questions with a gold-query label, {n_ok} reach an answer in the '
              f'subgraph ({n_empty} do not -> excluded, train.py falls back to sp_paths), '
              f'mean trails/question {sum(len(d["paths"]) for d in full) / max(1, n_ok):.1f}')


if __name__ == '__main__':
    main()
