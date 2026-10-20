"""Inference for REL-RAG (paper App. B "Inference") + retrieval metrics + export for reason/main.py.

Path-based: sample an initial question triple from softmax(<z_q, z_topic>) over V_cand, expand by
sampling the next triple among line-graph out-neighbours (+ STOP) with the Eq. 8 scorer until STOP or
max_depth; paths are ranked by probability and their triples added until the budget is reached.
Triple-based: rank all triples by <z_q, z_v> and keep the top-k.
"""
import os, json, argparse, math
import numpy as np
import torch
from tqdm import tqdm

from dataset import LineGraphDataset, load_embeddings
from model import RelRagRetriever


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


@torch.no_grad()
def decode_paths(out, s, budget=50, n_start=100, n_expand=3, max_depth=4, seed=0):
    rng = np.random.default_rng(seed)
    z = out['z'].float().cpu().numpy()
    zq = out['zq'].float().cpu().numpy()
    stop = out['stop'].float().cpu().numpy()
    cand = s.topic_cand.numpy()
    if len(cand) == 0:
        return [], []
    ptr, idx = s.nbr_ptr.numpy(), s.nbr_idx.numpy()
    p0 = _softmax(out['z_topic'][s.topic_cand.to(out['z'].device)].float().cpu().numpy() @ zq)
    starts = np.unique(rng.choice(len(cand), size=n_start, replace=True, p=p0))

    paths = {}

    def expand(ci, greedy):
        node = int(cand[ci])
        path, logp = [node], math.log(max(p0[ci], 1e-15))
        ctx = zq + z[node]
        for _ in range(max_depth):
            nbrs = idx[ptr[node]:ptr[node + 1]]
            logits = np.concatenate([z[nbrs] @ ctx, [stop @ ctx]]) if len(nbrs) else np.array([stop @ ctx])
            pr = _softmax(logits)
            k = int(pr.argmax()) if greedy else int(rng.choice(len(pr), p=pr))
            logp += math.log(max(pr[k], 1e-15))
            if k == len(nbrs):
                break
            node = int(nbrs[k])
            if node in path:           # avoid cycles
                break
            path.append(node)
            ctx = ctx + z[node]
        key = tuple(path)
        paths[key] = max(paths.get(key, -1e9), logp)

    for ci in starts:
        expand(ci, greedy=True)
        for _ in range(n_expand):
            expand(ci, greedy=False)
    # keep sampling until the retrieved paths cover the triple budget (App. B)
    for _ in range(10):
        if len({k for p in paths for k in p}) >= budget:
            break
        for ci in rng.choice(len(cand), size=n_start, replace=True, p=p0):
            expand(ci, greedy=False)

    ranked = sorted(paths.items(), key=lambda kv: -kv[1])
    triples, seen, kept_paths = [], set(), []
    for p, lp in ranked:
        if len(triples) >= budget:
            break
        kept_paths.append((list(p), lp))
        for k in p:
            if k not in seen:
                seen.add(k); triples.append((k, lp))
    return triples[:budget], kept_paths


@torch.no_grad()
def decode_best_first(out, s, budget=50, max_depth=4, max_pops=20000):
    """Enumerate paths in decreasing probability (best-first search over the Eq. 8 policy) and add
    their triples until the budget is filled; sampling-based decode_paths often stops short of it."""
    import heapq
    z = out['z'].float().cpu().numpy()
    zq = out['zq'].float().cpu().numpy()
    stop = out['stop'].float().cpu().numpy()
    cand = s.topic_cand.numpy()
    if len(cand) == 0:
        return [], []
    ptr, idx = s.nbr_ptr.numpy(), s.nbr_idx.numpy()
    p0 = _softmax(out['z_topic'][s.topic_cand.to(out['z'].device)].float().cpu().numpy() @ zq)
    # heap items: (-logp, path, context, finished)
    heap = [(-math.log(max(p, 1e-15)), (int(c),), None, False) for c, p in zip(cand, p0)]
    heapq.heapify(heap)
    triples, seen, kept_paths, pops = [], set(), [], 0
    while heap and len(triples) < budget and pops < max_pops:
        nlp, path, ctx, done = heapq.heappop(heap); pops += 1
        if done:
            kept_paths.append((list(path), -nlp))
            for k in path:
                if k not in seen:
                    seen.add(k); triples.append((k, -nlp))
            continue
        node = path[-1]
        ctx = (zq + z[path[0]]) if ctx is None else ctx
        nbrs = [n for n in idx[ptr[node]:ptr[node + 1]] if n not in path] if len(path) < max_depth else []
        logits = np.concatenate([z[nbrs] @ ctx, [stop @ ctx]]) if nbrs else np.array([stop @ ctx])
        lp = np.log(np.maximum(_softmax(logits), 1e-15))
        heapq.heappush(heap, (nlp - lp[-1], path, ctx, True))               # STOP here
        for j in np.argsort(-lp[:-1])[:20]:                                  # top-20 continuations
            n = int(nbrs[j])
            heapq.heappush(heap, (nlp - lp[j], path + (n,), ctx + z[n], False))
    return triples[:budget], kept_paths


@torch.no_grad()
def decode_reanchor(out, s, budget=50, B=3, max_depth=4):
    """Anchor-equivariant re-anchoring (docs/09) applied to REL-RAG P, no retraining. Pass 1: best-first from the
    topic triples. Bridges: entities of pass 1's top-B triples other than the topic entities. Pass 2: best-first with
    start candidates = triples whose head is a topic or a bridge. Fusion: max of z-normalised triple scores."""
    t1, p1 = decode_best_first(out, s, budget=budget, max_depth=max_depth)
    topics = {int(s.h[k]) for k in s.topic_cand.tolist()}
    bridges = []
    for k, _ in t1[:B]:
        for e in (int(s.h[k]), int(s.t[k])):
            if e not in topics and e not in bridges:
                bridges.append(e)
    if not bridges:
        return t1, p1
    orig = s.topic_cand
    s.topic_cand = torch.from_numpy(np.nonzero(np.isin(s.h, sorted(topics | set(bridges))))[0].astype(np.int64))
    try:
        t2, p2 = decode_best_first(out, s, budget=budget, max_depth=max_depth)
    finally:
        s.topic_cand = orig

    def z(lst):
        v = np.array([sc for _, sc in lst], dtype=np.float64)
        return {k: (sc - v.mean()) / (v.std() + 1e-9) for k, sc in lst} if len(v) else {}
    fused = z(t1)
    for k, sc in z(t2).items():
        fused[k] = max(fused.get(k, -1e9), sc)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:budget], p1 + p2


@torch.no_grad()
def decode_authors(out, s, K=800, N=3, M=1200, max_depth=6, min_prob=1e-4, seed=0, leak=None):
    """No triple budget: RAPL Trainerv3.decoding(way='normal') with the K=800, N=3, M=1200 of model_inference.py.
    K topic triples are drawn with replacement from softmax(<z_q, z_topic>); from each distinct one, N paths are
    sampled step by step (out-neighbours + STOP) until STOP or max_depth=6 (Trainer default); a path's probability is
    the product of its step probabilities (the topic probability is not included, as in compute_path_prob); the
    top-M paths with probability >= 1e-4 are kept. Their triples, in path-probability order, are the retrieval, so
    its size is set by STOP and the 1e-4 cut, not by a budget.
    Deviation: the authors also add the ground-truth topic triples (topic_labels) to the start set at test time;
    that uses test labels, so only the sampled topic triples are used here, unless `leak` (line-graph node ids, see
    --leak_topic) is given to measure that effect (EXP-31, P1)."""
    rng = np.random.default_rng(seed)
    z = out['z'].float().cpu().numpy()
    zq = out['zq'].float().cpu().numpy()
    stop = out['stop'].float().cpu().numpy()
    cand = s.topic_cand.numpy()
    if len(cand) == 0:
        return [], []
    ptr, idx = s.nbr_ptr.numpy(), s.nbr_idx.numpy()
    p0 = _softmax(out['z_topic'][s.topic_cand.to(out['z'].device)].float().cpu().numpy() @ zq)
    starts = np.unique(rng.choice(len(cand), size=K, replace=True, p=p0))
    if leak is not None and len(leak):           # way='normal': sampled topic triples UNION label-based topic triples
        pos = {int(c): j for j, c in enumerate(cand)}
        starts = np.union1d(starts, [pos[int(k)] for k in leak if int(k) in pos]).astype(np.int64)
    paths = []
    for ci in starts:
        for _ in range(N):
            node = int(cand[ci])
            path, logp = [node], 0.0
            ctx = zq + z[node]
            for _ in range(max_depth):
                nbrs = idx[ptr[node]:ptr[node + 1]]
                pr = _softmax(np.concatenate([z[nbrs] @ ctx, [stop @ ctx]]))
                k = int(rng.choice(len(pr), p=pr))
                logp += math.log(max(pr[k], 1e-15))
                if k == len(nbrs):
                    break
                node = int(nbrs[k])
                path.append(node)
                ctx = ctx + z[node]
            paths.append((path, logp))
    paths.sort(key=lambda x: -x[1])
    best = {}
    for p, lp in paths[:M]:
        if lp >= math.log(min_prob):
            best.setdefault(tuple(p), lp)   # transform_and_deduplicate_paths
    triples, seen = [], set()
    for p, lp in best.items():
        for k in p:
            if k not in seen:
                seen.add(k); triples.append((k, lp))
    return triples, [(list(p), lp) for p, lp in best.items()]


@torch.no_grad()
def rank_triples(out, budget):
    scores = (out['z'] @ out['zq']).float().cpu()
    k = min(budget, scores.numel())
    top = torch.topk(scores, k)
    return list(zip(top.indices.tolist(), top.values.tolist()))


@torch.no_grad()
def hybrid(out, s, budget, path_frac=0.5, **kw):
    """Improvement: path-decoded triples first (precision), then fill with triple ranking (diversity)."""
    scored, paths = decode_paths(out, s, budget=max(1, int(budget * path_frac)), **kw)
    seen = {k for k, _ in scored}
    for k, sc in rank_triples(out, budget + len(seen)):
        if len(scored) >= budget:
            break
        if k not in seen:
            seen.add(k); scored.append((k, sc - 1e6))   # ranked after all path triples
    return scored, paths


def retrieval_metrics(nodes, s):
    rec = s.rec
    A = set(rec['a_entity_id_list'])
    m = {}
    if A:
        ents = {int(s.h[k]) for k in nodes} | {int(s.t[k]) for k in nodes}
        m['ans_recall'] = len(A & ents) / len(A)
        m['ans_hit'] = float(bool(A & ents))
    labels = rec.get('paths') or rec['sp_paths']
    if labels:
        ns = set(nodes)
        m['label_path_hit'] = float(any(all(k in ns for k in p) for p in labels))
        lt = {k for p in labels for k in p}
        m['label_triple_recall'] = len(lt & ns) / len(lt)
    m['n_triples'] = len(nodes)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--dataset', default='webqsp', help='test dataset (models are trained on WebQSP)')
    ap.add_argument('--mode', choices=['path', 'path_bf', 'path_bf_reanchor', 'path_free', 'triple', 'hybrid'], required=True)
    ap.add_argument('--budgets', default='50')
    ap.add_argument('--residual', action='store_true')
    ap.add_argument('--no_self_loops', action='store_true')
    ap.add_argument('--n_start', type=int, default=100)
    ap.add_argument('--n_expand', type=int, default=3)
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--path_frac', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--shift_anchor', action='store_true', help='E1: start from a neighbour of the topic instead of the topic')
    ap.add_argument('--reanchor_B', type=int, default=3, help='path_bf_reanchor: bridges from the top-B pass-1 triples')
    ap.add_argument('--leak_topic', choices=['none', 'sp', 'sp_minlen'], default='none',
                    help='path_free only (EXP-31, P1): add the first triple of the test shortest paths q->answer to the '
                         'start set, as the authors\' decoding(way="normal") adds topic_labels; sp_minlen = shortest only')
    args = ap.parse_args()
    dev = torch.device(os.environ.get('DEV', 'cuda:0'))   # DEV=cpu when the local GPU is taken

    # Model config comes from the training run's args.json when present, so architecture flags cannot mismatch.
    cfg_path = os.path.join(os.path.dirname(args.ckpt), 'args.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    model = RelRagRetriever(self_loops=not cfg.get('no_self_loops', args.no_self_loops),
                            residual=cfg.get('residual', args.residual),
                            bidirectional=not cfg.get('no_bidirectional', False),
                            combine=cfg.get('combine', 'mean'), gnn_dropout=bool(cfg.get('gnn_dropout', 1))).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    emb = load_embeddings(args.split, args.dataset)
    ds = LineGraphDataset(args.split, emb, dataset=args.dataset)
    # path_free has no budget: one result set, saved as retrieval_result_free.pth
    budgets = ['free'] if args.mode == 'path_free' else [int(b) for b in args.budgets.split(',')]
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    metrics = {b: [] for b in budgets}
    export = {b: {} for b in budgets}
    for i in tqdm(range(n), desc=f'{args.split}:{args.mode}'):
        s = ds[i]
        try:
            out = model(s, dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model_cpu = model.to('cpu'); out = model_cpu(s, torch.device('cpu')); model.to(dev)
        rec = s.rec
        names, rels = rec['entity_names'], rec['relation_list']
        if args.shift_anchor:
            # E1 intervention (docs/09): the start candidates become the triples headed by one neighbour e' of a topic
            # entity (seeded choice among text entities that are neither topics nor answers); question unchanged
            topics = set(rec['q_entity_id_list']); A = set(rec['a_entity_id_list'])
            nb = sorted({int(v) for h, t in zip(s.h, s.t) for u, v in ((h, t), (t, h))
                         if int(u) in topics and int(v) < rec['num_text_entities'] and int(v) not in topics and int(v) not in A})
            if not nb:
                continue
            e1 = nb[np.random.default_rng(i).integers(len(nb))]
            s.topic_cand = torch.from_numpy(np.nonzero(s.h == e1)[0].astype(np.int64))
            if len(s.topic_cand) == 0:
                continue
        for b in budgets:
            if args.mode == 'path':
                scored, paths = decode_paths(out, s, budget=b, n_start=args.n_start,
                                             n_expand=args.n_expand, max_depth=args.max_depth, seed=i)
            elif args.mode == 'path_free':
                leak = None
                if args.leak_topic != 'none' and rec['sp_paths']:
                    sp = rec['sp_paths']
                    if args.leak_topic == 'sp_minlen':
                        sp = [p for p in sp if len(p) == min(map(len, sp))]
                    leak = sorted({p[0] for p in sp})
                scored, paths = decode_authors(out, s, seed=i, leak=leak)
            elif args.mode == 'path_bf_reanchor':
                scored, paths = decode_reanchor(out, s, budget=b, B=args.reanchor_B, max_depth=args.max_depth)
            elif args.mode == 'path_bf':
                scored, paths = decode_best_first(out, s, budget=b, max_depth=args.max_depth)
            elif args.mode == 'hybrid':
                scored, paths = hybrid(out, s, budget=b, path_frac=args.path_frac, n_start=args.n_start,
                                       n_expand=args.n_expand, max_depth=args.max_depth, seed=i)
            else:
                scored, paths = rank_triples(out, b), None
            nodes = [k for k, _ in scored]
            metrics[b].append(retrieval_metrics(nodes, s))
            export[b][rec['id']] = {
                'question': rec['question'], 'a_entity': rec['a_entity'],
                'scored_triples': [(names[s.h[k]], rels[s.r[k]], names[s.t[k]], float(sc)) for k, sc in scored],
                'paths': None if paths is None else [
                    ' -> '.join([names[s.h[p[0]]]] + [f'{rels[s.r[k]]} -> {names[s.t[k]]}' for k in p])
                    for p, _ in paths],
            }
    out_dir = os.path.join(os.path.dirname(args.ckpt), f'retrieval_{"" if args.dataset == "webqsp" else args.dataset + "_"}{args.split}_{args.mode}{args.tag}')
    os.makedirs(out_dir, exist_ok=True)
    for b in budgets:
        agg = {}
        for key in ['ans_recall', 'ans_hit', 'label_path_hit', 'label_triple_recall', 'n_triples']:
            vals = [m[key] for m in metrics[b] if key in m]
            agg[key] = float(np.mean(vals)) if vals else None
        print(f'budget={b}: ' + ', '.join(f'{k}={v:.4f}' for k, v in agg.items() if v is not None))
        torch.save(export[b], os.path.join(out_dir, f'retrieval_result_{b}.pth'))
        with open(os.path.join(out_dir, f'metrics_{b}.txt'), 'w') as f:
            f.write(str(agg) + '\n')
    print('saved to', out_dir)


if __name__ == '__main__':
    main()
