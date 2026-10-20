"""Test-time retrieval of a trained run on EVERY test question, in the format of retrieve/eval.py.

Modes (--modes, comma-separated; all are inference only, no retraining):
  base     one pass from the question's topic entities
  aer      anchor-equivariant re-anchoring: pass 1 from the topics; the entities of its top-3 triples (other than the
           topics) become extra anchors ("bridges") for pass 2
  adr      anchor decomposition: one pass from all topics + one pass from each topic alone (only differs from base on
           questions with >= 2 topic entities)
  adr_aer  both: every anchor set of adr, each followed by its own re-anchoring pass
Several passes are fused per triple by the max of their z-normalised scores (a pass can only lift a triple).
RBF runs are scored with the complete-path max-marginal (mm) switched on, as in every reported RBF number.

Questions: data/<ds>/test.pkl (answer in the subgraph) + data/<ds>/test_extra.pkl (the other test questions), i.e. the
full test set (WebQSP 1,638 of 1,639: one question has no subgraph; CWQ 3,531).

  python infer.py --run webqsp_wl_gte_sparql2_s42 --dataset cwq --modes base,aer,adr,adr_aer
Writes runs/<run>/pred_<dataset>_<mode>.pth = {question id: {'question', 'q_entity', 'a_entity_in_graph',
'scored_triples': [(head, relation, tail, score)] top 500}} -> evaluate with retrieve/eval.py.
"""
import argparse, json, os
import torch
from tqdm import tqdm

from train_retriever import ROOT, attach_bge_prior, build_scorer, load_split, to_device

RUNS = f'{ROOT}/exp_qrdde/runs'
MODES = ['base', 'aer', 'adr', 'adr_aer']
TOP_K = 500
N_BRIDGE_TRIPLES = 3                       # AER: bridges = entities of the top-3 triples of a pass


def z_normalise(scores):
    return (scores - scores.mean()) / (scores.std() + 1e-6)


class QuestionScorer:
    """scores one question under every mode; each anchor set is scored once and shared by the modes"""
    def __init__(self, model, rec, emb, device):
        self.model, self.rec, self.emb, self.device = model, rec, emb, device
        self.topics = sorted(set(rec['q_ent']))
        self.cache = {}

    def pass_from(self, anchors):
        """logits of one pass with `anchors` as topic entities"""
        key = tuple(sorted(anchors))
        if key not in self.cache:
            self.cache[key] = self.model(to_device(self.rec, self.emb, self.device, anchors=key))
        return self.cache[key]

    def bridges(self, anchors, logits):
        """entities of the top-N_BRIDGE_TRIPLES triples, other than the anchors"""
        top = torch.topk(logits, min(N_BRIDGE_TRIPLES, len(logits))).indices.cpu().numpy()
        return (set(self.rec['h'][top].tolist()) | set(self.rec['t'][top].tolist())) - set(anchors)

    def with_reanchoring(self, anchors):
        """[pass from anchors] + [pass from anchors + bridges] when there are bridges"""
        first = self.pass_from(anchors)
        new = self.bridges(anchors, first)
        return [first] + ([self.pass_from(set(anchors) | new)] if new else [])

    def scores(self, mode):
        single = [{t} for t in self.topics] if len(self.topics) > 1 else []
        if mode == 'base':
            passes = [self.pass_from(self.topics)]
        elif mode == 'aer':
            passes = self.with_reanchoring(self.topics)
        elif mode == 'adr':
            passes = [self.pass_from(self.topics)] + [self.pass_from(s) for s in single]
        else:                                                    # adr_aer
            passes = self.with_reanchoring(self.topics) + [p for s in single for p in self.with_reanchoring(s)]
        return passes[0] if len(passes) == 1 else torch.stack([z_normalise(p) for p in passes]).max(0).values


def main():
    ap = argparse.ArgumentParser(description='Retrieve top-500 triples for every test question (see the docstring).')
    ap.add_argument('--run', required=True, help='run directory name in exp_qrdde/runs (or a path)')
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'], help='test set to retrieve on')
    ap.add_argument('--modes', default='base', help=f'comma-separated, from {MODES}')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    modes = args.modes.split(',')
    assert set(modes) <= set(MODES), f'unknown mode in {modes}'
    run_dir = args.run if os.path.isdir(args.run) else f'{RUNS}/{args.run}'
    train_args = json.load(open(f'{run_dir}/result.json'))['args']
    model = build_scorer(train_args)
    model.load_state_dict(torch.load(f'{run_dir}/best.pt', map_location=args.device))
    model.to(args.device).eval()
    if hasattr(model, 'rbf'):
        model.rbf.mm = True                                      # complete-path scoring at test time
    questions = []
    for split in ('test', 'test_extra'):
        records, emb = load_split(args.dataset, split)
        attach_bge_prior(records, args.dataset, split, train_args.get('rel_prior', 'none'))
        questions += [(rec, emb) for rec in records]
    out = {mode: {} for mode in modes}
    with torch.no_grad():
        for rec, emb in tqdm(questions, desc=f'{os.path.basename(run_dir)} on {args.dataset}'):
            scorer = QuestionScorer(model, rec, emb, args.device)
            names, relations = rec['entity_names'], rec['relation_list']
            for mode in modes:
                scores = scorer.scores(mode)
                top = torch.topk(scores, min(TOP_K, len(scores))).indices.tolist(); scores = scores.cpu()
                out[mode][rec['id']] = {
                    'question': rec['question'], 'q_entity': [names[i] for i in rec['q_ent']],
                    'a_entity_in_graph': [names[i] for i in rec.get('a_ent', [])],
                    'scored_triples': [(names[rec['h'][i]], relations[rec['r'][i]], names[rec['t'][i]], float(scores[i])) for i in top]}
    for mode in modes:
        path = f'{run_dir}/pred_{args.dataset}_{mode}.pth'
        torch.save(out[mode], path); print(f'{len(out[mode])} questions -> {path}')


if __name__ == '__main__':
    main()
