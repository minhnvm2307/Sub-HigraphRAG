"""Train a triple retriever on WebQSP or CWQ (full data, GTE-large text embeddings), SubgraphRAG protocol: one BCE over
the triples of each question, Adam lr 1e-3, model selection on val triple_recall@100, patience 10, seed 42.

Retrievers (--feat):
  dde          SubgraphRAG: MLP over [z_q ; z_h + DDE(h) ; z_r ; z_t + DDE(t)], DDE = topic one-hot + 2 forward + 2
               reverse rounds of directional distance encoding
  dde+rt       RT: + a DDE whose edges are weighted by a relation-topic score s(r) = 0.5 lexical(q, r) + 0.5 sem(q, r)
               (floor 0.1) and a one-hot "r is in the question's top-5 relations" feature on every triple.
               sem = <z_r, z_q> with GTE (--rt_src gte) or the fine-tuned BGE cosine (--rt_src bge, needs --rel_prior)
               WL = RT trained with --pos_weight effective_num (per-question positive weight E_neg / E_pos,
               E_n = (1 - 0.999^n) / (1 - 0.999))
  dde+ase, dde+rt+ase
               ASE: + anchor-set encoding (a DDE per topic entity, embedded by a shared MLP, pooled mean / max / min)
  rbfonly      RBF-Ret (rbf.py): step-factorised max-product paths from the topic entities; no SubgraphRAG MLP.
               Trained with non-backtracking walks; switch on the complete-path max-marginal at test time (infer.py).
Labels (--labels): sp = SubgraphRAG shortest paths (default); v1 / v2 / v3 = gold-query labels (labels.py), questions
without a gold-query label keep sp. Applied to train AND val, so model selection follows the same target.
BGE relation prior (--rel_prior bge | bge2): relscore_bge.py scores of the BGE-M3 LoRA relation encoder tuned on
WebQSP train (bge: shortest-path positives; bge2: SPARQL v2 positives) or on a 5k subset of CWQ train (bge_cwq: SPARQL v2
positives, bge_sparql_data.py --max-questions 5000), read by RBF (an input of pi) and by RT with
--rt_src bge.

  python train_retriever.py --dataset webqsp --feat dde+rt --pos_weight effective_num --labels v2 --out runs/webqsp_wl_v2
  python train_retriever.py --dataset cwq --feat rbfonly --labels v2 --device cpu --out runs/cwq_rbf_v2
Writes <out>/best.pt and <out>/result.json ({'args', 'log': one entry per epoch, 'best_epoch'}). Test retrieval and
evaluation are separate: infer.py, then retrieve/eval.py. --resume <out>/best.pt --start_epoch N continues a run.
"""
import argparse, json, os, pickle, random, sys, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

ROOT = os.environ.get('SGRAG_RETRIEVE', '/home/minhnv/Lab/SubgraphRAG/retrieve')   # override on other machines
RELRAG = '/home/minhnv/Lab/SubgraphRAG/relrag'                                        # holds the WebQSP / CWQ memmaps
DATA = f'{ROOT}/exp_qrdde/data'
RUNS = f'{ROOT}/exp_qrdde/runs'
EXTERNAL = {'grailqa_paper': '/home/minhnv/Lab/SubgraphRAG/data_ext/grailqa_paper'}   # test-only sets (old ext_data.py layout)
sys.path.insert(0, ROOT)
from src.model.retriever import DDE
from src.dataset.retriever import ShardedEmbDict, compute_lexical_score, clean_text
from src.setup import set_seed
from rbf import RBF, log_to_logit

FEATURES = ['dde', 'dde+rt', 'dde+ase', 'dde+rt+ase', 'rbfonly']
LABEL_FILES = {'v1': 'sparql', 'v2': 'sparqlv2', 'v3': 'sparqlv3'}                   # data/<ds>/<file>_<split>.pkl
BGE_SCORE_DIRS = {'bge': 'scores_ft', 'bge2': 'scores_ft_sparql2', 'bge_cwq': 'scores_cwq5k'}   # data/bge_p11/<dir>/<ds>_<split>.pkl
EMB, ASE_DIM, DDE_ROUNDS, RT_TOP_RELATIONS, RT_FLOOR = 1024, 64, 2, 5, 0.1
VAL_K = [20, 50, 100, 200]


# ============================================================ data: records, embeddings, labels, priors
class MemmapEmb:
    """per-question embeddings from one flat memmap directory (q.npy, ent.npy, rel.npy + index.pkl)"""
    def __init__(self, d):
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))
        self.q, self.ent, self.rel = (np.load(f'{d}/{n}.npy', mmap_mode='r') for n in ('q', 'ent', 'rel'))

    def get(self, qid):
        k, e0, ne, r0, nr = self.index[qid]
        return (torch.from_numpy(np.array(self.q[k:k + 1])), torch.from_numpy(np.array(self.ent[e0:e0 + ne])),
                torch.from_numpy(np.array(self.rel[r0:r0 + nr])))


class GlobalEmb:
    """per-question embeddings as rows of shared tables (q.npy, ent_table.npy, rel_table.npy + index.pkl): the compact
    copy of CWQ train / val (data/cwq/gemb_<split>, ~4 GB instead of ~43 GB of shards)"""
    def __init__(self, d):
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))
        self.q, self.ent, self.rel = (np.load(f'{d}/{n}.npy', mmap_mode='r') for n in ('q', 'ent_table', 'rel_table'))

    def get(self, qid):
        k, entity_rows, relation_rows = self.index[qid]
        return (torch.from_numpy(np.array(self.q[k:k + 1])), torch.from_numpy(np.asarray(self.ent[entity_rows])),
                torch.from_numpy(np.asarray(self.rel[relation_rows])))


class ShardEmb:
    """per-question embeddings read from the original GTE shard files (CWQ train / val when no compact copy exists)"""
    def __init__(self, shard_dir, max_cache_shards=4):
        self.src = ShardedEmbDict(shard_dir, json.load(open(f'{shard_dir}/shard_index.json')), max_cache_shards=max_cache_shards)

    def get(self, qid):
        e = self.src[qid]
        return e['q_emb'], e['entity_embs'], e['relation_embs']

    def shard_of(self, qid):
        return self.src._index.get(str(qid), '')


class RelationOnlyEmb:
    """question + relation vectors only (data/<ds>/lemb_<split>): RBF never reads entity vectors, only how many text
    entities there are, so its training loads ~0.15 GB instead of ~4 GB. Entity matrix = empty (n_text, 0)."""
    def __init__(self, d):
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))                 # qid -> (q row, relation rows, n_text)
        self.q, self.rel = np.load(f'{d}/q.npy'), np.load(f'{d}/rel_table.npy')

    def get(self, qid):
        k, relation_rows, n_text = self.index[qid]
        return torch.from_numpy(self.q[k:k + 1]), torch.empty((n_text, 0)), torch.from_numpy(self.rel[relation_rows])


def load_split(dataset, split, relation_only=False):
    """-> (records, embedding store). Records: data/<ds>/<split>.pkl ('test' = questions with an answer in the graph;
    'test_extra' = the other test questions). relation_only: RBF training tables (also sets rec['lex'])."""
    if dataset in EXTERNAL:                                    # one test file, global GTE tables in <dir>/emb
        if split != 'test':
            return [], None
        records = pickle.load(open(f'{EXTERNAL[dataset]}/test.pkl', 'rb'))
        for rec in records:                                    # field names of the main datasets
            rec['entity_names'], rec['relation_list'] = rec.pop('names'), rec.pop('rels')
            rec['label'] = np.zeros(len(rec['h']), np.float32)
        return records, GlobalEmb(f'{EXTERNAL[dataset]}/emb')
    records = pickle.load(open(f'{DATA}/{dataset}/{split}.pkl', 'rb'))
    split_dir = 'test' if split == 'test_extra' else split                   # test_extra shares the test embeddings
    if relation_only:
        d = f'{DATA}/{dataset}/lemb_{split_dir}'
        lexical_scores = pickle.load(open(f'{d}/lex.pkl', 'rb'))
        for rec in records:
            rec['lex'] = lexical_scores[rec['id']]
        return records, RelationOnlyEmb(d)
    if os.path.exists(f'{DATA}/{dataset}/gemb_{split_dir}/index.pkl'):
        return records, GlobalEmb(f'{DATA}/{dataset}/gemb_{split_dir}')
    if dataset == 'webqsp':
        return records, MemmapEmb(f'{RELRAG}/data/emb_memmap/{split_dir}')
    if split_dir == 'test':
        return records, MemmapEmb(f'{RELRAG}/data/cwq/emb_memmap/test')
    return records, ShardEmb(f'{ROOT}/data_files/cwq/emb/gte-large-en-v1.5')


def apply_gold_query_labels(records, dataset, split, version):
    """replace the shortest-path label of each record by its gold-query label (chain + constraint triples) when one
    exists; the other records keep the shortest-path label"""
    labels = pickle.load(open(f'{DATA}/{dataset}/{LABEL_FILES[version]}_{split}.pkl', 'rb'))
    n_relabelled = 0
    for rec in records:
        label = labels.get(rec['id'])
        if label and label['chain']:
            new = np.zeros_like(rec['label']); new[label['chain'] + label['cons']] = 1
            rec['label'] = new; rec['chain_idx'] = np.array(label['chain']); n_relabelled += 1
    print(f'[labels {version}] {dataset} {split}: {n_relabelled}/{len(records)} records relabelled, '
          f'{len(records) - n_relabelled} keep the shortest-path label', flush=True)


def attach_bge_prior(records, dataset, split, kind):
    """rec['rel_prior'] = [n_relations, 2]: (BGE cosine of question and relation, cosine - the question's max cosine);
    zeros for a question without a cached score"""
    if kind == 'none':
        return
    raw = pickle.load(open(f'{DATA}/bge_p11/{BGE_SCORE_DIRS[kind]}/{dataset}_{split}.pkl', 'rb'))
    n_missing = 0
    for rec in records:
        scores = raw.get(str(rec['id']))
        if scores is None or len(scores) != len(rec['relation_list']):
            rec['rel_prior'] = np.zeros((len(rec['relation_list']), 2), np.float32); n_missing += 1
        else:
            rec['rel_prior'] = np.stack([scores, scores - scores.max()], 1).astype(np.float32)
    print(f'[prior {kind}] {dataset} {split}: {len(records)} records, {n_missing} without a cached score', flush=True)


def lexical(rec):
    """WL's lexical relation score (src/dataset/retriever.py), cached on the record"""
    if 'lex' not in rec:
        words = clean_text(rec['question'])
        rec['lex'] = np.array([compute_lexical_score(words, rel) for rel in rec['relation_list']], dtype=np.float32)
    return rec['lex']


def to_device(rec, emb, device, anchors=None):
    """model input of one question; anchors: entity ids marked as topic entities (default: the question's topics)"""
    q, ent, rel = emb.get(rec['id'])
    topic = torch.zeros(rec['n_ent'], dtype=torch.long); topic[sorted(anchors) if anchors is not None else rec['q_ent']] = 1
    as_long = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)
    return {'h': as_long(rec['h']), 'r': as_long(rec['r']), 't': as_long(rec['t']), 'n_ent': rec['n_ent'],
            'q': q.float().to(device), 'ent': ent.float().to(device), 'rel': rel.float().to(device),
            'topic': F.one_hot(topic, 2).float().to(device), 'lex': torch.from_numpy(lexical(rec)).to(device),
            'label': torch.from_numpy(rec['label'].astype(np.float32)).to(device),
            'rel_prior': torch.from_numpy(rec['rel_prior']).to(device) if 'rel_prior' in rec else None,
            **stop_targets(rec, len(rel), device)}


def stop_targets(rec, n_rel, device):
    """targets of the stop factor phi, [n_relations, 2] (forward / reverse last step):
    stop_label = 1 for the last step of a gold walk, i.e. a chain triple (shortest-path triple without a gold-query
    label) whose far end is a gold answer; stop_mask = 1 where such a last step exists at all (a step of that relation
    and direction into a text entity, i.e. a place a walk may stop)"""
    h, r, t, n_text = rec['h'], rec['r'], rec['t'], rec['n_text']
    answers = np.zeros(rec['n_ent'], dtype=bool); answers[list(rec.get('a_ent', []))] = True
    on_chain = rec['chain_idx'] if 'chain_idx' in rec else np.nonzero(rec['label'])[0]
    label, mask = np.zeros((n_rel, 2), np.float32), np.zeros((n_rel, 2), np.float32)
    mask[r[t < n_text], 0] = 1; mask[r[h < n_text], 1] = 1
    label[r[on_chain][answers[t[on_chain]]], 0] = 1; label[r[on_chain][answers[h[on_chain]]], 1] = 1
    as_t = lambda a: torch.from_numpy(a).to(device)
    return {'stop_label': as_t(label), 'stop_mask': as_t(mask)}


def epoch_order(records, emb):
    """shuffled training order; with shard files, shuffle shards then questions inside a shard (a flat shuffle would
    reload a shard on almost every question)"""
    if not isinstance(emb, ShardEmb):
        order = records[:]; random.shuffle(order); return order
    groups = defaultdict(list)
    for rec in records:
        groups[emb.shard_of(rec['id'])].append(rec)
    keys = list(groups); random.shuffle(keys)
    return [rec for k in keys for rec in random.sample(groups[k], len(groups[k]))]


# ============================================================ model
class Scorer(nn.Module):
    """one logit per triple; module names are those of the checkpoints in runs/ (do not rename)"""
    def __init__(self, feat, rel_prior='none', rt_src='gte', pi_input='full', stop=False):
        super().__init__()
        self.feat, self.rt_src = feat, rt_src
        parts = feat.split('+')
        if 'rbfonly' in parts:
            self.rbf = RBF(EMB, extra_prior=2 if rel_prior != 'none' else 0, pi_input=pi_input, stop=stop)
            return
        self.non_text = nn.Embedding(1, EMB)                    # shared vector of entities without text (CVT nodes)
        entity_dim = EMB + 2                                     # text vector + topic one-hot
        if 'dde' in parts:
            self.dde = DDE(num_rounds=DDE_ROUNDS, num_reverse_rounds=DDE_ROUNDS); entity_dim += 4 * DDE_ROUNDS
        if 'ase' in parts:
            self.ase_dde = DDE(num_rounds=DDE_ROUNDS, num_reverse_rounds=DDE_ROUNDS)
            self.ase_phi = nn.Sequential(nn.Linear(2 + 4 * DDE_ROUNDS, ASE_DIM), nn.ReLU(), nn.Linear(ASE_DIM, ASE_DIM))
            entity_dim += 3 * ASE_DIM
        triple_extra = 0
        if 'rt' in parts:
            self.rt_dde = DDE(num_rounds=DDE_ROUNDS, num_reverse_rounds=DDE_ROUNDS); entity_dim += 4 * DDE_ROUNDS
            triple_extra += 2
        self.pred = nn.Sequential(nn.Linear(2 * EMB + 2 * entity_dim + triple_extra, EMB), nn.ReLU(), nn.Linear(EMB, 1))

    def forward(self, x):
        if hasattr(self, 'rbf'):
            return log_to_logit(self.rbf(x) - 0.6931471805599453)          # final - log 2 = union for one anchor
        edges = torch.stack([x['h'], x['t']])
        non_text = self.non_text.weight.expand(x['n_ent'] - len(x['ent']), -1)
        entity = [torch.cat([x['ent'], non_text]), x['topic']]
        if hasattr(self, 'dde'):
            entity += self.dde(x['topic'], edges, edges.flip(0))
        if hasattr(self, 'ase_dde'):                                         # one DDE per anchor, pooled over anchors
            per_anchor = []
            for a in x['topic'][:, 1].nonzero().reshape(-1).tolist():
                one_hot = torch.zeros(x['n_ent'], 2, device=edges.device); one_hot[:, 0] = 1; one_hot[a, 0] = 0; one_hot[a, 1] = 1
                per_anchor.append(self.ase_phi(torch.cat([one_hot] + self.ase_dde(one_hot, edges, edges.flip(0)), 1)))
            P = torch.stack(per_anchor) if per_anchor else torch.zeros(1, x['n_ent'], ASE_DIM, device=edges.device)
            entity.append(torch.cat([P.mean(0), P.max(0).values, P.min(0).values], 1))
        triple_extra = []
        if hasattr(self, 'rt_dde'):
            semantic = x['rel_prior'][:, 0] if self.rt_src == 'bge' else (x['rel'] @ x['q'].T).squeeze(-1)
            relation_score = 0.5 * x['lex'] + 0.5 * semantic
            in_top = torch.zeros_like(relation_score)
            in_top[torch.topk(relation_score, min(RT_TOP_RELATIONS, len(relation_score))).indices] = 1
            entity += self.rt_dde(x['topic'], edges, edges.flip(0), edge_weight=torch.clamp(relation_score[x['r']], min=RT_FLOOR))
            triple_extra.append(F.one_hot(in_top[x['r']].long(), 2).float())
        entity = torch.cat(entity, 1)
        features = torch.cat([x['q'].expand(len(x['r']), -1), entity[x['h']], x['rel'][x['r']], entity[x['t']]] + triple_extra, 1)
        return self.pred(features).view(-1)


def build_scorer(args):
    """Scorer from the 'args' of a result.json (or this script's arguments)"""
    return Scorer(args['feat'], rel_prior=args.get('rel_prior', 'none'), rt_src=args.get('rt_src', 'gte'),
                  pi_input=args.get('pi_input', 'full'), stop=bool(args.get('stop', False)))


# ============================================================ validation + training
@torch.no_grad()
def validate(model, records, emb, device):
    """answer recall and triple recall (against the training labels) at VAL_K"""
    model.eval()
    metrics = defaultdict(list)
    stop_head = getattr(getattr(model, 'rbf', None), 'stop', None) is not None
    for rec in records:
        x = to_device(rec, emb, device)
        if stop_head and x['stop_label'].sum() > 0:             # the most likely last step is a gold one
            logits = model.rbf.stop_logits(x).masked_fill(x['stop_mask'] == 0, -1e9)
            metrics['stop_top1'].append(float(x['stop_label'].flatten()[logits.argmax()] > 0))
        order = torch.argsort(model(x), descending=True).cpu().numpy()
        positive = rec['label'].astype(bool)
        for k in VAL_K:
            top = order[:k]
            entities = set(rec['h'][top].tolist()) | set(rec['t'][top].tolist())
            metrics[f'ans_recall@{k}'].append(len(set(rec['a_ent']) & entities) / len(rec['a_ent']))
            if positive.any():
                metrics[f'triple_recall@{k}'].append(positive[top].sum() / positive.sum())
    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def stop_loss(model, x):
    """BCE of the stop factor over the possible last steps of the question, positives and negatives weighted
    equally (a question has a few gold last steps among tens of relations); 0 without a gold last step"""
    mask, label = x['stop_mask'] > 0, x['stop_label']
    if not (label[mask] > 0).any():
        return 0.0
    bce = F.binary_cross_entropy_with_logits(model.rbf.stop_logits(x), label, reduction='none')
    pos, neg = mask & (label > 0), mask & (label == 0)
    return 0.5 * bce[pos].mean() + (0.5 * bce[neg].mean() if neg.any() else 0.0)


def effective_number_weight(labels):
    """WL loss: positive weight E_neg / E_pos with E_n = (1 - 0.999^n) / (1 - 0.999); None without positives"""
    n_pos = int(labels.sum().item()); n_neg = len(labels) - n_pos
    if n_pos == 0:
        return None
    effective = lambda n: (1.0 - 0.999 ** n) / (1.0 - 0.999)
    return torch.tensor(effective(n_neg) / (effective(n_pos) + 1e-8), device=labels.device)


def main():
    ap = argparse.ArgumentParser(description='Train a triple retriever (see the module docstring).')
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'])
    ap.add_argument('--feat', required=True, choices=FEATURES)
    ap.add_argument('--labels', default='sp', choices=['sp', 'v1', 'v2', 'v3'], help='training labels (labels.py)')
    ap.add_argument('--pos_weight', default='none', choices=['none', 'effective_num'], help="'effective_num' = WL's loss")
    ap.add_argument('--rel_prior', default='none', choices=['none', 'bge', 'bge2', 'bge_cwq'], help='BGE relation prior (RBF input / RT score)')
    ap.add_argument('--rt_src', default='gte', choices=['gte', 'bge'], help="semantic half of RT's relation score")
    ap.add_argument('--pi_input', default='full', choices=['full', 'no_lex', 'no_prod', 'no_lex_prod'],
                    help='rbfonly: ablation of the inputs of pi(r | q) (rbf.py)')
    ap.add_argument('--stop', action='store_true', help='rbfonly: add the stop factor phi(stop | q, r) (rbf.py), BCE on the gold last steps')
    ap.add_argument('--init_from', default=None, help='--stop only: load pi from this run (exp_qrdde/runs/<run>), freeze it and train phi alone (model selection on val stop_top1)')
    ap.add_argument('--train_eval', type=int, default=300, help='also log the val metrics on this many fixed train questions (train vs val curves); 0 = off')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0', help="'cpu' is enough for rbfonly")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10000)
    ap.add_argument('--patience', type=int, default=10, help='stop after this many epochs without a better val triple_recall@100')
    ap.add_argument('--resume', default=None, help='continue from these weights (Adam restarts from zero moments)')
    ap.add_argument('--start_epoch', type=int, default=0, help='number of the first resumed epoch')
    args = ap.parse_args()
    assert args.rt_src == 'gte' or args.rel_prior != 'none', '--rt_src bge reads the BGE scores of --rel_prior'
    set_seed(args.seed, deterministic=True)                     # as retrieve/train.py; before any CUDA call
    device = torch.device(args.device); os.makedirs(args.out, exist_ok=True)
    relation_only = args.feat == 'rbfonly'
    train, emb_train = load_split(args.dataset, 'train', relation_only)
    val, emb_val = load_split(args.dataset, 'val', relation_only)
    for records, split in ((train, 'train'), (val, 'val')):
        if args.labels != 'sp':
            apply_gold_query_labels(records, args.dataset, split, args.labels)
        attach_bge_prior(records, args.dataset, split, args.rel_prior)
    model = build_scorer(vars(args)).to(device)
    if args.init_from:                                          # stop factor only: pi frozen at a trained run
        missing, _ = model.load_state_dict(torch.load(f'{RUNS}/{args.init_from}/best.pt', map_location=device), strict=False)
        assert all(k.startswith('rbf.stop') for k in missing), missing
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith('rbf.stop')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    select = 'stop_top1' if args.init_from else 'triple_recall@100'
    train_probe = random.Random(0).sample(train, min(args.train_eval, len(train))) if args.train_eval else []
    # wandb as retrieve/train.py (project REL-RAG); WANDB_MODE=disabled turns it off
    wandb.init(project='REL-RAG', name=f'{os.path.basename(os.path.normpath(args.out))}_{time.strftime("%b%d-%H:%M:%S", time.gmtime())}',
               config=vars(args))
    best, epochs_without_gain, log = -1, 0, []
    if args.resume:                                             # the resumed weights are the current best
        model.load_state_dict(torch.load(args.resume, map_location=device))
        v = validate(model, val, emb_val, device)
        best = v[select]; torch.save(model.state_dict(), f'{args.out}/best.pt')
        log.append({'epoch': args.start_epoch - 1, 'resumed_from': args.resume, **{f'val_{k}': x for k, x in v.items()}})
    for epoch in range(args.start_epoch, args.epochs):
        t0, total_loss = time.time(), 0.0
        for rec in tqdm(epoch_order(train, emb_train), desc=f'epoch {epoch}'):
            x = to_device(rec, emb_train, device)
            loss = stop_loss(model, x) if args.stop else 0.0
            if not args.init_from:
                weight = effective_number_weight(x['label']) if args.pos_weight == 'effective_num' else None
                loss = loss + F.binary_cross_entropy_with_logits(model(x), x['label'], pos_weight=weight)
            if not torch.is_tensor(loss):
                continue
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
        v = validate(model, val, emb_val, device)
        vt = validate(model, train_probe, emb_train, device) if train_probe else {}    # same metrics on train questions
        log.append({'epoch': epoch, 'loss': total_loss / len(train), 'sec': time.time() - t0, **{f'val_{k}': x for k, x in v.items()},
                    **{f'train_{k}': x for k, x in vt.items()}})
        improved = v[select] > best
        if improved:
            best, epochs_without_gain = v[select], 0
            torch.save(model.state_dict(), f'{args.out}/best.pt')
        else:
            epochs_without_gain += 1
        print(f'epoch {epoch}: loss {total_loss / len(train):.4f}  val {select} {v[select]:.4f}  '
              f'ans_recall@50 {v["ans_recall@50"]:.4f}' + ('  (best, saved)' if improved else f'  ({epochs_without_gain} without gain)'), flush=True)
        wandb.log({'epoch': epoch, 'loss': total_loss / len(train), **{f'val/{k}': x for k, x in v.items()},
                   **{f'train/{k}': x for k, x in vt.items()}})
        if epochs_without_gain >= args.patience:
            break
    best_epoch = max(log, key=lambda e: e[f'val_{select}'])['epoch']
    json.dump({'args': vars(args), 'best_epoch': best_epoch, 'log': log}, open(f'{args.out}/result.json', 'w'), indent=1)
    print(json.dumps({'run': args.out, 'best_epoch': best_epoch}))


if __name__ == '__main__':
    main()
