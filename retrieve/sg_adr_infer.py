"""ADR / AER inference for SubgraphRAG-pipeline checkpoints (retrieve/train.py, e.g. RT-combined), no retraining.

Same fusion as exp_qrdde/adr_probe.py: each pass rescored with a different anchor set (the topic one-hot fed to
DDE), passes fused by the max of z-scores.
  base     all topics                                   aer      all topics, then + bridges of the top-B triples
  adr      all topics + each topic alone                adr_aer  aer on all topics + aer on each topic alone

  python sg_adr_infer.py -p webqsp_rt_combined_<ts>/cpt.pth [--config_file configs/retriever/transfer/..._to_cwq.yaml] \
      --tag cwq --modes base,aer,adr,adr_aer
Writes <ckpt dir>/adr_<tag>_<mode>.pth (top-500, same format as inference.py's retrieval_result.pth).
"""
import os, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.config.retriever import load_yaml, strip_listwise
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

z = lambda v: (v - v.mean()) / (v.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True)
    ap.add_argument('--config_file', default=None)
    ap.add_argument('--tag', required=True, help='output tag, e.g. webqsp or cwq')
    ap.add_argument('--modes', default='base,aer,adr,adr_aer')
    ap.add_argument('--B', type=int, default=3)
    ap.add_argument('--max_K', type=int, default=500)
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    cpt = torch.load(args.path, map_location='cpu')
    config = strip_listwise(load_yaml(args.config_file) if args.config_file else cpt['config'])
    set_seed(config['env']['seed'], deterministic=config['env'].get('deterministic', True))
    ds = RetrieverDataset(config=config, split='test', skip_no_path=False)
    model = Retriever(ds[0]['q_emb'].shape[-1], **config['retriever']).to(dev)
    model.load_state_dict(cpt['model_state_dict']); model.eval()
    modes = args.modes.split(',')
    out = {m: {} for m in modes}

    def run_one(raw, dev):
        (h, r, t, q_emb, ent, n_nt, rel, topic_oh, _, _, rtf) = prepare_sample(dev, collate_retriever([raw]))
        q_emb, ent, rel, rtf = q_emb.float(), ent.float(), rel.float(), rtf.to(dev)   # external sets store fp16
        names = raw['text_entity_list'] + raw['non_text_entity_list']; rels = raw['relation_list']
        if len(h) == 0:
            for m in modes:
                out[m][raw['id']] = {'question': raw['question'], 'scored_triples': [],
                                     'a_entity_in_graph': [names[e] for e in raw['a_entity_id_list']]}
            return
        n = topic_oh.shape[0]; cache = {}
        def S(anchors):
            key = tuple(sorted(anchors))
            if key not in cache:
                mask = torch.zeros(n, dtype=torch.long, device=dev); mask[list(key)] = 1
                cache[key] = model(h, r, t, q_emb, ent, n_nt, rel, F.one_hot(mask, 2).float(), rtf).reshape(-1)
            return cache[key]
        def chain(anchors):
            s = S(anchors)
            top = torch.topk(s, min(args.B, len(s))).indices
            new = (set(h[top].tolist()) | set(t[top].tolist())) - set(anchors)
            return [s] + ([S(set(anchors) | new)] if new else [])
        tops = set(topic_oh[:, 1].nonzero().reshape(-1).tolist())
        single = [{x} for x in sorted(tops)] if len(tops) > 1 else []
        for m in modes:
            passes = {'base': lambda: [S(tops)], 'aer': lambda: chain(tops),
                      'adr': lambda: [S(tops)] + [S(x) for x in single],
                      'adr_aer': lambda: chain(tops) + [p for x in single for p in chain(x)]}[m]()
            sc = passes[0] if len(passes) == 1 else torch.stack([z(p) for p in passes]).max(0).values
            top = torch.topk(sc, min(args.max_K, len(sc)))
            out[m][raw['id']] = {
                'question': raw['question'],
                'scored_triples': [(names[h[j].item()], rels[r[j].item()], names[t[j].item()], float(v))
                                   for v, j in zip(top.values.tolist(), top.indices.tolist())],
                'a_entity_in_graph': [names[e] for e in raw['a_entity_id_list']]}

    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            raw = ds[i]
            try:
                run_one(raw, dev)
            except torch.cuda.OutOfMemoryError:     # large external subgraphs on a shared GPU: this question on CPU
                torch.cuda.empty_cache(); model.cpu(); run_one(raw, torch.device('cpu')); model.to(dev)
    d = os.path.dirname(args.path)
    for m in modes:
        torch.save(out[m], os.path.join(d, f'adr_{args.tag}_{m}.pth'))
    print('done', args.path, args.tag, modes)


if __name__ == '__main__':
    main()
