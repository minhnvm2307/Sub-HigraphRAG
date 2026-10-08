"""Train the REL-RAG retriever (paper Sec. 4, App. B-C, F.1).

--regime path   : L = L_q (Eq. 11, initial triple) + L_path (Eq. 8, next-triple softmax with STOP)
--regime triple : L = L_triple (Eq. 12), positives = triples on label paths, negatives 1:5
Setup from F.1: Adam lr 1e-3, batch size 10 (gradient accumulation over graphs), 15 epochs,
dropout 0.2, no batch norm.
Model selection (--select): the paper says "validation loss", but the authors' code
(RAPL/src/model/Trainerv3.py::fit) validates only for epoch > 5 and keeps checkpoints by the F1 between the end
triples of decoded paths and the end triples of the validation label paths ('path_f1', default for path/joint).
The triple regime is not in the authors' code, so it keeps validation loss.
"""
import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LineGraphDataset, load_embeddings, collate_single
from model import RelRagRetriever
from infer import decode_paths, rank_triples, hybrid, retrieval_metrics


def topic_loss(out, s, dev, mode='all'):
    """Eq. 11 over V_cand; positives from Eq. 10 (s.topic_pos).
    mode 'all': every candidate, as Eq. 11 is written. mode 'sample': as the authors' Trainerv3.train_step --
    one random positive and up to 8 random negatives per graph, skipped when there are fewer than 3 negatives."""
    cand = s.topic_cand.to(dev)
    if cand.numel() == 0 or s.topic_pos.numel() == 0:
        return None
    is_pos = torch.isin(cand, s.topic_pos.to(dev))
    if mode == 'sample':
        pos, neg = cand[is_pos], cand[~is_pos]
        if pos.numel() == 0 or neg.numel() < 3:
            return None
        p = pos[random.randrange(pos.numel())]
        n = neg[torch.randperm(neg.numel(), device=dev)[:8]]
        zq = out['zq']
        return (F.binary_cross_entropy_with_logits((out['z_topic'][p] @ zq).view(1), torch.ones(1, device=dev))
                + F.binary_cross_entropy_with_logits(out['z_topic'][n] @ zq, torch.zeros(n.numel(), device=dev)))
    logits = out['z_topic'][cand] @ out['zq']
    return F.binary_cross_entropy_with_logits(logits, is_pos.float())


def path_loss(out, s, dev):
    """Eq. 8: autoregressive next-triple prediction over out-neighbours + STOP, one random label path."""
    if not s.paths:
        return None
    p = random.choice(s.paths)
    z, zq, stop = out['z'], out['zq'], out['stop']
    ptr, idx = s.nbr_ptr, s.nbr_idx
    ctx = zq + z[p[0]]
    losses = []
    # steps 1..K-1 predict the next label triple, the final step predicts STOP
    for step in range(1, len(p) + 1):
        cur = p[step - 1]
        nbrs = idx[ptr[cur]:ptr[cur + 1]]
        logits = torch.cat([z[nbrs.to(dev)] @ ctx, (stop @ ctx).view(1)])
        if step < len(p):
            hit = (nbrs == p[step]).nonzero()
            assert hit.numel() > 0, 'label path is not a line-graph walk'
            target = int(hit[0])
        else:
            target = len(nbrs)
        losses.append(F.cross_entropy(logits.view(1, -1), torch.tensor([target], device=dev)))
        if step < len(p):
            ctx = ctx + z[p[step]]
    return torch.stack(losses).mean()


def triple_loss(out, s, dev, neg_ratio=5):
    """Eq. 12 with positives = triples on label paths and 1:5 random negatives (F.1)."""
    if not s.paths:
        return None
    n = out['z'].shape[0]
    pos = torch.tensor(sorted({k for p in s.paths for k in p}), device=dev)
    mask = torch.ones(n, dtype=torch.bool, device=dev); mask[pos] = False
    neg_pool = mask.nonzero().view(-1)
    k = min(neg_ratio * pos.numel(), neg_pool.numel())
    neg = neg_pool[torch.randperm(neg_pool.numel(), device=dev)[:k]]
    idx = torch.cat([pos, neg])
    logits = out['z'][idx] @ out['zq']
    target = torch.cat([torch.ones(pos.numel(), device=dev), torch.zeros(k, device=dev)])
    return F.binary_cross_entropy_with_logits(logits, target)


def sample_loss(model, s, dev, regime, topic_mode='all'):
    out = model(s, dev)
    if regime == 'path':
        parts = [l for l in (topic_loss(out, s, dev, topic_mode), path_loss(out, s, dev)) if l is not None]
    elif regime == 'joint':   # improvement: both objectives on one encoder
        parts = [l for l in (topic_loss(out, s, dev, topic_mode), path_loss(out, s, dev), triple_loss(out, s, dev)) if l is not None]
    else:
        parts = [l for l in (triple_loss(out, s, dev),) if l is not None]
    return (sum(parts) if parts else None), out


@torch.no_grad()
def validate(model, loader, dev, regime, budget):
    model.eval()
    losses, mets = [], []
    tp = n_pred = n_true = 0          # end-triple F1 as in the authors' Trainerv3.fit / cond_evaluate
    for i, s in enumerate(loader):
        try:
            loss, out = sample_loss(model, s, dev, regime)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        if loss is not None:
            losses.append(loss.item())
        paths = None
        if regime == 'path':
            scored, paths = decode_paths(out, s, budget=budget, seed=i)
        elif regime == 'joint':
            scored, paths = hybrid(out, s, budget=budget, seed=i)
        else:
            scored = rank_triples(out, budget)
        mets.append(retrieval_metrics([k for k, _ in scored], s))
        if paths is not None and s.paths:
            pred, gold = {p[-1] for p, _ in paths}, {p[-1] for p in s.paths}
            tp += len(pred & gold); n_pred += len(pred); n_true += len(gold)
    model.train()
    agg = {k: float(np.mean([m[k] for m in mets if k in m])) for k in ['ans_recall', 'ans_hit', 'label_path_hit']}
    prec, rec = tp / max(n_pred, 1), tp / max(n_true, 1)
    agg['path_f1'] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return float(np.mean(losses)), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=['path', 'triple', 'joint'], required=True)
    ap.add_argument('--labels', default='sp', help="'sp' or 'paths:<train.jsonl>' (merge_labels.py)")
    ap.add_argument('--val_labels', default='sp', help="validation labels; the authors annotate train only")
    ap.add_argument('--tau', type=float, default=0.95)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--residual', action='store_true')
    ap.add_argument('--no_self_loops', action='store_true', help='original RAPL GCNConv(add_self_loops=False)')
    ap.add_argument('--no_bidirectional', action='store_true')
    ap.add_argument('--val_budget', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--select', choices=['path_f1', 'val_loss'], default=None,
                    help="default: path_f1 (authors' code) for path/joint, val_loss for triple")
    ap.add_argument('--select_after', type=int, default=5, help='path_f1 selection only considers epoch > this')
    ap.add_argument('--combine', choices=['mean', 'sum'], default='mean', help="fwd/bwd GCN: paper App. Eq. 86 'mean'; authors' code 'sum'")
    ap.add_argument('--gnn_dropout', type=int, default=1, help="dropout between GCN layers (authors' code: 0)")
    ap.add_argument('--topic_loss', choices=['all', 'sample'], default='all', help="Eq. 11 as written ('all') or the authors' 1 pos + <=8 neg ('sample')")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'), indent=2)
    dev = torch.device('cuda:0')
    val_budget = args.val_budget or (500 if args.regime == 'triple' else 50)
    select = args.select or ('val_loss' if args.regime == 'triple' else 'path_f1')

    train_labels, val_labels = args.labels, args.val_labels
    emb_tr = load_embeddings('train')
    train_ds = LineGraphDataset('train', emb_tr, train_labels, args.tau, require_label=True)
    emb_va = load_embeddings('val')
    val_ds = LineGraphDataset('val', emb_va, val_labels, args.tau, require_label=True)
    print('train label stats', train_ds.label_stats, '| val', val_ds.label_stats)
    print(f'train={len(train_ds)} val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single, num_workers=0)

    model = RelRagRetriever(dropout=args.dropout, self_loops=not args.no_self_loops,
                            bidirectional=not args.no_bidirectional, residual=args.residual,
                            combine=args.combine, gnn_dropout=bool(args.gnn_dropout)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log, best = [], -float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        tot, cnt, skipped, acc = 0.0, 0, 0, 0
        opt.zero_grad()
        for s in tqdm(train_loader, desc=f'epoch {epoch}', mininterval=30):
            try:
                loss, _ = sample_loss(model, s, dev, args.regime, args.topic_loss)
                if loss is None:
                    continue
                (loss / args.batch_size).backward()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(); torch.cuda.empty_cache(); skipped += 1; acc = 0
                continue
            tot += loss.item(); cnt += 1; acc += 1
            if acc == args.batch_size:
                opt.step(); opt.zero_grad(); acc = 0
        if acc:
            opt.step(); opt.zero_grad()
        val_loss, val_m = validate(model, val_loader, dev, args.regime, val_budget)
        row = {'epoch': epoch, 'train_loss': tot / max(cnt, 1), 'val_loss': val_loss, 'oom_skipped': skipped,
               'minutes': (time.time() - t0) / 60, **{f'val_{k}@{val_budget}': v for k, v in val_m.items()}}
        log.append(row); print(json.dumps(row), flush=True)
        json.dump(log, open(f'{args.out}/log.json', 'w'), indent=2)
        torch.save(model.state_dict(), f'{args.out}/epoch_{epoch}.pt')
        score = -val_loss if select == 'val_loss' else val_m['path_f1']
        if select == 'path_f1' and epoch <= args.select_after:
            continue
        if score > best:
            best = score
            torch.save(model.state_dict(), f'{args.out}/best.pt')
            json.dump({'epoch': epoch, 'select': select, 'score': score}, open(f'{args.out}/best.json', 'w'))
    print('best val loss' if select == 'val_loss' else 'best val path_f1', abs(best) if select == 'val_loss' else best)


if __name__ == '__main__':
    main()
