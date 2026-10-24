"""Phase 11 BGE-M3 LoRA training data with gold-query (SPARQL v2) relation positives (10-20).
Runs the Phase 11 builder (branch phase11-bge-finetune: retrieve/build_relation_finetune_dataset.py + phase11/data.py)
unchanged except for the positive set: for a question with a v2 label (make_sparql_v2.py: chain trails ending at a gold
answer + their constraint triples) the positive triples are those, instead of the edge-complete shortest paths; questions
without a v2 label keep the shortest paths, as qrdde.apply_sparql_labels does. Negatives, the entity-disjoint fit/dev split
and the record format are the builder's own, so the recipe v1_dense_entity_disjoint trains on the output as before.
  python bge_sparql_data.py --p11 <checkout with retrieve/phase11> --output-dir <dir> [--version v2] [--max-questions N]
         [builder args]
--max-questions N: build from a seeded random subset of N questions of the split (e.g. CWQ train, 27k questions, when the
tuning has to fit a small GPU); the comparison splits of the builder are not subsampled.
"""
import argparse, pickle, sys

ROOT = '/home/minhnv/Lab/SubgraphRAG'
D = f'{ROOT}/retrieve/exp_qrdde/data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p11', required=True); ap.add_argument('--version', default='v2')
    ap.add_argument('--max-questions', type=int, default=0, help='seeded random subset of the split (0 = all)')
    a, rest = ap.parse_known_args()
    sys.path.insert(0, f'{a.p11}/retrieve')
    import phase11.data as P
    import build_relation_finetune_dataset as B
    ds = rest[rest.index('--dataset') + 1] if '--dataset' in rest else 'webqsp'
    split = rest[rest.index('--source-split') + 1] if '--source-split' in rest else 'train'
    tag = {'v2': 'sparqlv2', 'v3': 'sparqlv3'}[a.version]
    lab = pickle.load(open(f'{D}/{ds}/{tag}_{split}.pkl', 'rb'))
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/{split}.pkl', 'rb'))}
    sp_derive, used = P.derive_edge_complete_shortest_paths, {'sparql': 0, 'sp': 0}

    def derive(sample, graph):
        qid = str(sample['id']); l = lab.get(qid)
        if not l or not l['chain'] or qid not in recs:
            used['sp'] += 1
            return sp_derive(sample, graph)
        rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
        H, R, T = rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist()
        ix = {}
        for k, (h, r, t) in enumerate(zip(graph.heads, graph.relation_ids, graph.tails)):
            ix.setdefault((graph.entity_values[h], graph.relation_values[r], graph.entity_values[t]), []).append(k)
        triples = set()
        for j in l['chain'] + l['cons']:
            key = (N[H[j]], L[R[j]], N[T[j]])
            assert key in ix, f'{qid}: label triple {key} not in the Phase 11 sample graph'
            triples.update(ix[key])
        used['sparql'] += 1
        pairs = {(graph.heads[k], graph.tails[k]) for k in triples}
        rels = {graph.relation_values[graph.relation_ids[k]] for k in triples}
        return [], triples, rels, pairs

    P.derive_edge_complete_shortest_paths = derive
    if a.max_questions:                                        # subsample the processed split the builder reads
        import random
        load_all, target = B._load_pickle, f'/processed/{split}.pkl'
        def load_subset(path):
            samples = load_all(path)
            if str(path).endswith(target) and f'/{ds}/' in str(path):
                samples = random.Random(42).sample(samples, min(a.max_questions, len(samples)))
                print(f'{ds} {split}: random subset of {len(samples)} questions (seed 42)', flush=True)
            return samples
        B._load_pickle = load_subset
    out = B.main(rest)
    print(f'{ds} {split}: positives from {tag} for {used["sparql"]} questions, shortest paths for {used["sp"]}', flush=True)
    return out


if __name__ == '__main__':
    raise SystemExit(main())
