"""Chunk-max relation-topic scores (EXP-30): a parameter-free variant of RT-combined's semantic channel.

RT-combined scores a relation by cos(z_question, z_relation) on the whole question. Multi-constraint questions (CWQ
conjunction / composition, 85-92% of CWQ) ask about several relations, each matching only part of the question, so the
whole-question embedding dilutes them. Here the semantic score is the max over question chunks (sliding word windows
plus the full question) of cos(z_chunk, z_relation); the lexical channel, the hybrid mix (alpha 0.5) and the top-5
mask are unchanged, so the only difference to RT-combined is the semantic channel. Same encoder and pooling as the
stored GTE embeddings (gte-large-en-v1.5, CLS, L2-normalised).

  python precompute_chunk_rt.py --dataset webqsp --splits train,val,test
Writes data_files/<ds>/relation_topics/chunkmax/<split>_hybrid_alpha0.5_top5.pth in the relation-topic cache format
(retrieve/src/dataset/retriever.py::_load_relation_topic_cache).
"""
import argparse, os, pickle
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from src.dataset.retriever import compute_lexical_score, clean_text


def chunks(q, w=4, stride=2):
    words = q.split()
    out = [q] + [' '.join(words[i:i + w]) for i in range(0, max(len(words) - w, 0) + 1, stride)]
    return list(dict.fromkeys(c for c in out if c.strip()))


@torch.no_grad()
def encode(texts, tok, model, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
        out.append(F.normalize(model(**b).last_hidden_state[:, 0], p=2, dim=1).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--w', type=int, default=4)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, unpad_inputs=True,
                                      use_memory_efficient_attention=True).to('cuda').eval().half()
    out_dir = f'data_files/{args.dataset}/relation_topics/chunkmax'; os.makedirs(out_dir, exist_ok=True)
    for split in args.splits.split(','):
        recs = pickle.load(open(f'data_files/{args.dataset}/processed/{split}.pkl', 'rb'))
        rel_vocab = sorted({r for s in recs for r in s['relation_list']})
        R = encode(rel_vocab, tok, model); rid = {r: k for k, r in enumerate(rel_vocab)}
        feats = {}
        for n, s in enumerate(recs):
            if not s['relation_list']:
                continue
            C = encode(chunks(s['question'], args.w), tok, model)
            Z = R[[rid[r] for r in s['relation_list']]]
            semantic = (Z @ C.T).max(1).values
            lexical = torch.tensor([compute_lexical_score(clean_text(s['question']), r) for r in s['relation_list']])
            score = args.alpha * lexical + (1 - args.alpha) * semantic
            mask = torch.zeros_like(score); mask[torch.topk(score, min(args.topk, len(score))).indices] = 1
            feats[s['id']] = torch.stack([score, mask], 1).float()
            if n % 2000 == 0:
                print(split, n, len(recs), flush=True)
        payload = {'metadata': {'dataset': args.dataset, 'text_encoder_name': 'gte-large-en-v1.5', 'split': split,
                                'relation_topic_source': 'gte', 'relation_score_mode': 'hybrid',
                                'relation_score_alpha': float(args.alpha), 'use_relation_topic_mask': True,
                                'topk_rt': args.topk, 'feature_layout': ['score', 'mask'], 'variant': f'chunkmax_w{args.w}'},
                   'features': feats}
        torch.save(payload, f'{out_dir}/{split}_hybrid_alpha{args.alpha:g}_top{args.topk}.pth')
        print('saved', split, len(feats))


if __name__ == '__main__':
    main()
