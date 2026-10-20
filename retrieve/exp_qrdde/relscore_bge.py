"""Relation relevance from the Phase 11 recipe (BGE-M3 + LoRA, dense mode, trained on WebQSP train only; the original
checkpoint was lost, so it is retrained with retrieve/configs/phase11/v1_dense_entity_disjoint.json on vast).
Encodes every question (max 96 tokens) and every unique relation text ('split' format of phase11/relation_text.py:
'.' and '_' -> spaces; max 48 tokens) with CLS pooling + L2 normalisation (as FlagEmbedding's M3 dense training), and
writes the cosine of each question with each of its candidate relations.
Input: relq/<name>.jsonl rows {id, question, relations}. Output: relq_scores/<name>.pkl = {id: float32 [n_rel]}.
  python relscore_bge.py --adapter <run>/final --inputs relq --out relq_scores [--base] (base = no adapter)
"""
import argparse, json, os, pickle
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def split_relation(r):
    return r.replace('.', ' ').replace('_', ' ')


@torch.no_grad()
def encode(model, tok, texts, max_len, bs=128, dev='cuda'):
    out = []
    order = np.argsort([len(t) for t in texts])
    embs = [None] * len(texts)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = tok([texts[j] for j in idx], max_length=max_len, truncation=True, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            h = model(**b).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h.float(), dim=-1).cpu()
        for k, j in enumerate(idx):
            embs[j] = h[k]
    return torch.stack(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=None); ap.add_argument('--base', action='store_true')
    ap.add_argument('--inputs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='BAAI/bge-m3')
    ap.add_argument('--device', default='cuda', help="'cpu' runs fp32 (no autocast); used locally for a few extra questions")
    args = ap.parse_args()
    
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = torch.float16 if args.device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dt).to(args.device).eval()
    if not args.base:
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    os.makedirs(args.out, exist_ok=True)
    
    for fn in sorted(os.listdir(args.inputs)):
        if not fn.endswith('.jsonl'):
            continue
        rows = [json.loads(l) for l in open(os.path.join(args.inputs, fn))]
        rel_vocab = {}
        for r in rows:
            for rel in r['relations']:
                rel_vocab.setdefault(rel, len(rel_vocab))
        R = encode(model, tok, [split_relation(r) for r in rel_vocab], 48, dev=args.device)
        Q = encode(model, tok, [r['question'] for r in rows], 96, dev=args.device)
        out = {}
        for k, r in enumerate(rows):
            idx = torch.tensor([rel_vocab[x] for x in r['relations']], dtype=torch.long)
            out[r['id']] = (R[idx] @ Q[k]).numpy().astype(np.float32)
        pickle.dump(out, open(os.path.join(args.out, fn.replace('.jsonl', '.pkl')), 'wb'))
        print(fn, len(rows), 'questions,', len(rel_vocab), 'unique relations', flush=True)


if __name__ == '__main__':
    main()
