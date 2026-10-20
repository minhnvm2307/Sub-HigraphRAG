"""E1 (WEB_RESEARCH.md): QR-DDE vs DDE inside the SubgraphRAG pointwise MLP retriever, on small subsets
or (--full) on the complete WebQSP/CWQ training data (M2, docs/01_RESEARCH_PLAN.md).

--feat dde        : SubgraphRAG as is (topic one-hot + DDE 2 fwd + 2 reverse rounds)
--feat qrdde      : DDE replaced by Query-conditioned Relational DDE (reduced NBFNet):
                    h^0 = 1[v in topic] * u_q,
                    h^k = h^{k-1} + ReLU(LN(W_k(h^0 + MEAN_{(u,r,v)} h_u^{k-1} * w_q(r)))),
                    w_q(r) = MLP([z_r ; z_q]), separate outputs for the forward and inverse direction.
--feat qrdde_noq  : QR-DDE ablation, w(r) = MLP([z_r]) only (no query-conditioning).
--feat rtdde      : DDE edge-weighted by a static (non-learned) relation-text relevance score
                    S(q,r) = cos(z_q, z_r), clamped at a floor -- no propagation MLP, no query
                    conditioning during message passing. Legacy "RT-DDE" concept from
                    docs/archive/subgraphrag/PIPELINE_PRECOMPUTE_AND_EXPORT.md, reinterpreted here
                    to compute S(q,r) on the fly from the same GTE embeddings (no precompute step).
--feat dde+qrdde  : any '+'-joined combination of the above feature sets.
Loss, scorer MLP and model selection (val triple_recall@100) follow retrieve/train.py.

Anchor-equivariant retrieval (AER, docs/09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md), any --feat:
--infer_passes 2  : pass 1 scores from the topic entities; the entities of its top-B triples (other than the topics)
                    become extra anchors ("bridges") for pass 2; the final score of a triple is the max of its
                    z-normalised pass-1 and pass-2 scores (monotone: extra anchors never push a pass-1 triple down).
--reanchor_p p    : during training, with probability p, train on pass 2 (topic + the model's own bridges from a
                    no-grad pass 1) with the unchanged labels, so the scorer sees at train time the anchor sets it
                    meets at inference time (anchors are all treated alike: equivariance, no bridge channel).
feat 'rt'         : the WL relation-topic branch of SubgraphRAG-pipeline retrievers (src/model/retriever.py, dde_mode
                    'both' + use_relation_topic_mask; configs .../webqsp_wl_effective_num.yaml): relation score
                    s_r = 0.5 lexical + 0.5 <z_r, z_q> (hybrid, alpha 0.5), a DDE weighted by max(s_r, 0.1) per edge,
                    and a one-hot top-5 relation mask on each triple. 'dde+rt' = WL's feature set.
--pos_weight effective_num : WL's loss (src/loss.py): per-question BCE pos_weight = E_neg / E_pos,
                    E_n = (1 - beta^n) / (1 - beta), beta 0.999.
--anchor_aug_p p  : EXP-35, with probability p train a one-topic question with one extra anchor from
                    data/<dataset>/anchor_aug.pkl (make_anchor_aug.py: matched to CWQ second topics) and add
                    that anchor's SubgraphRAG shortest-path labels.
"""
import os, sys, json, time, pickle, random, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

ROOT = os.environ.get('SGRAG_RETRIEVE', '/home/minhnv/Lab/SubgraphRAG/retrieve')   # override on other machines
RELRAG = '/home/minhnv/Lab/SubgraphRAG/relrag'   # REL-RAG reproduction (moved out of RAPL/ on 10-07)
sys.path.insert(0, ROOT)
from src.model.retriever import DDE
from src.dataset.retriever import ShardedEmbDict, compute_lexical_score, clean_text
from src.setup import set_seed
from rbf import RBF, log_to_logit, NEG as NEG_RBF

K_LIST = [20, 50, 100, 200]
REL_PRIOR_DIM = {'none': 0, 'pmh': 2, 'mem': 2, 'bge': 2, 'bge2': 2}


def answer_softmax_loss(o, x):
    """listwise answer loss on the RBF best-path scores (node_logp [A, n], log space): per anchor,
    -log( sum_{v in answers} exp F_a(v) / sum_{v text, v not an anchor} exp F_a(v) ); anchors that reach no answer within
    T steps are skipped (no gradient through unreachable answers). Trains pi to rank the answers' best paths above every
    other entity's, which is what a top-k budget needs (BCE on triples is dominated by easy negatives)."""
    lp, ans = o['node_logp'], x['a_ent']
    if len(ans) == 0:
        return lp.sum() * 0
    n_text = len(x['ent']); anchors = x['topic'][:, 1].nonzero().reshape(-1)
    cand = torch.ones(lp.shape[1], dtype=torch.bool, device=lp.device); cand[n_text:] = False; cand[anchors] = False
    ans = ans[cand[ans]]
    if len(ans) == 0:
        return lp.sum() * 0
    pos, den = lp[:, ans], lp[:, cand]
    ok = pos.max(1).values > NEG_RBF / 2
    if not ok.any():
        return lp.sum() * 0
    return (torch.logsumexp(den[ok], 1) - torch.logsumexp(pos[ok], 1)).mean()


def attach_rel_prior(recs, dataset, split, kind, source=None):
    """per-relation prior columns on each record (rec['rel_prior'], [n_rel, d]); 'pmh' = the Path+Memory Hybrid relation
    selector of the SubgraphRAG work (retrieve/data_files/<ds>/relation_topics/path_memory_hybrid/<split>.pth: col 0 =
    fused path score in [0, 1], col 1 = selected-relation mask). Its memory uses the SAME dataset's training labels, so
    it is valid in-distribution only. Missing questions get zeros."""
    if kind == 'none':
        return recs
    if kind == 'mem':        # make_memory_prior.py: votes of the SOURCE (= training) dataset's train questions; valid on transfer
        cache = pickle.load(open(f'{ROOT}/exp_qrdde/data/{dataset}/memprior_{source}_{split}.pkl', 'rb'))
        cache = {k: torch.from_numpy(v) for k, v in cache.items()}
    elif kind in ('bge', 'bge2'):   # relscore_bge.py: Phase 11 BGE-M3 LoRA relation encoder retrained on WebQSP train only
        # ('bge2': positives from SPARQL v2 labels, bge_sparql_data.py); columns [cos(q, r), cos - max cos of the question]
        sub = {'bge': 'scores_ft', 'bge2': 'scores_ft_sparql2'}[kind]
        raw = pickle.load(open(f'{ROOT}/exp_qrdde/data/bge_p11/{sub}/{dataset}_{split}.pkl', 'rb'))
        cache = {k: torch.from_numpy(np.stack([v, v - v.max()], 1)) for k, v in raw.items()}
    else:
        cache = torch.load(f'{ROOT}/data_files/{dataset}/relation_topics/path_memory_hybrid/{split}.pth', weights_only=False)
    miss = 0
    for rec in recs:
        c = cache.get(str(rec['id']))
        if c is None or c.shape[0] != len(rec['relation_list']):
            c = torch.zeros((len(rec['relation_list']), 2)); miss += 1
        rec['rel_prior'] = c.float().numpy()
    print(f'[rel_prior {kind}] {dataset} {split}: {len(recs)} records, {miss} without a cache entry', flush=True)
    return recs


class SubsetEmb:
    def __init__(self, d):
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))
        self.q, self.ent, self.rel = (np.load(f'{d}/{n}.npy', mmap_mode='r') for n in ('q', 'ent', 'rel'))

    def get(self, sid):
        k, e0, ne, r0, nr = self.index[sid]
        return (torch.from_numpy(np.array(self.q[k:k + 1])), torch.from_numpy(np.array(self.ent[e0:e0 + ne])),
                torch.from_numpy(np.array(self.rel[r0:r0 + nr])))


class ShardEmbAdapter:
    """Same .get(sid)->(q,ent,rel) contract as SubsetEmb, backed by ShardedEmbDict (mmap'd shard
    files) instead of a flat memmap -- lets to_device()/epoch_order() treat both sources identically.
    Used for CWQ full train/val, which are too large to memmap given the disk budget (docs/05)."""
    def __init__(self, shard_dir, shard_index, max_cache_shards=4):
        self.src = ShardedEmbDict(shard_dir, shard_index, max_cache_shards=max_cache_shards)

    def get(self, sid):
        e = self.src[sid]
        return e['q_emb'], e['entity_embs'], e['relation_embs']

    def shard_key(self, sid):
        return self.src._index.get(str(sid), '')


class LeanEmb:
    """Relation-only embeddings for RBF-Ret training (make_lean_emb.py): RBF reads the question and relation vectors and
    only the NUMBER of text entities (len(x['ent'])), so the per-question entity matrix (n_ent x 1024, the bulk of every
    epoch's IO and host->device copies) is replaced by an empty (n_text, 0) tensor. Same .get contract as SubsetEmb.
    Whole tables are held in RAM (CWQ train: ~0.15 GB instead of ~4 GB of entity vectors)."""
    def __init__(self, d):
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))                 # sid -> (q row, rel_table ids, n_text)
        self.q, self.rel = np.load(f'{d}/q.npy'), np.load(f'{d}/rel_table.npy')

    def get(self, sid):
        k, ri, nt = self.index[sid]
        return torch.from_numpy(self.q[k:k + 1]), torch.empty((nt, 0)), torch.from_numpy(self.rel[ri])


def load(dataset, split, data_dir=None, emb_dir=None, shard_index=None, max_cache_shards=4):
    """data_dir: base dir holding <split>.pkl (default exp_qrdde/data/<dataset>, the 800/1500-q
    subset). emb_dir: embedding source dir; defaults to <data_dir>/emb_<split> (subset layout).
    shard_index: if given, emb_dir is read via ShardEmbAdapter (sharded) instead of SubsetEmb
    (flat memmap) -- used only for CWQ full train/val."""
    d = data_dir or f'{ROOT}/exp_qrdde/data/{dataset}'
    recs = pickle.load(open(f'{d}/{split}.pkl', 'rb'))
    if shard_index is not None:
        emb = ShardEmbAdapter(emb_dir, shard_index, max_cache_shards=max_cache_shards)
    else:
        emb = SubsetEmb(emb_dir or f'{d}/emb_{split}')
    return recs, emb


def apply_sparql_labels(recs, dataset, split, with_constraints=True, version=''):
    """Replace the SP labels by the gold-query labels (label_compare.py -> data/<ds>/sparql_<split>.pkl: triples of the
    gold chain walked inside the subgraph, plus the bound constraint triples). Questions without a bound chain keep SP.
    Applied to train AND val, so the model-selection metric (val triple_recall@100) follows the same target."""
    # version 'v2' (make_sparql_v2.py -> sparqlv2_<split>.pkl): only the gold-chain trails that END AT A GOLD ANSWER, and their constraints
    # version 'v3' (make_sparql_v3.py): v2 + hops walked in reverse when needed + chain-anchored shortest bridges to the answer
    lab = pickle.load(open(f'{ROOT}/exp_qrdde/data/{dataset}/sparql{version}_{split}.pkl', 'rb'))
    n = 0
    for rec in recs:
        q = lab.get(rec['id'])
        if q and q['chain']:
            new = np.zeros_like(rec['label']); new[q['chain'] + (q['cons'] if with_constraints else [])] = 1
            rec['label'] = new; n += 1
    print(f'[labels sparql{version}{"" if with_constraints else "_chain"}] {dataset} {split}: {n}/{len(recs)} records relabelled, '
          f'{len(recs) - n} keep SP', flush=True)


def resolve(dataset, split, full, max_cache_shards=4, lean=False):
    """Full-data embedding wiring for --full (see docs/01_RESEARCH_PLAN.md M2): WebQSP always
    reuses /relrag/data/emb_memmap/<split> (id-for-id identical to the SubsetEmb/make_memmap.py
    layout, verified: 1639/1639 ids match on webqsp test); CWQ test reuses
    /relrag/data/cwq/emb_memmap/test; CWQ train/val read straight from the GTE shard files
    (no memmap -- would need ~43G, only ~15-27G disk is free)."""
    if not full:
        return load(dataset, split)
    data_dir = f'{ROOT}/exp_qrdde/data/{dataset}'
    if lean:                                                         # RBF training: make_lean_emb.py tables + lex
        d = f'{data_dir}/lemb_{split}'
        recs = pickle.load(open(f'{data_dir}/{split}.pkl', 'rb')); lex = pickle.load(open(f'{d}/lex.pkl', 'rb'))
        for rec in recs:
            rec['lex'] = lex[rec['id']]
        return recs, LeanEmb(d)
    if os.path.exists(f'{data_dir}/gemb_{split}/index.pkl'):       # compact unique-entity copy (make_global_emb.py)
        from ext_data import GlobalEmb
        return pickle.load(open(f'{data_dir}/{split}.pkl', 'rb')), GlobalEmb.from_dir(f'{data_dir}/gemb_{split}')
    if dataset == 'webqsp':
        return load(dataset, split, data_dir=data_dir, emb_dir=f'{RELRAG}/data/emb_memmap/{split}')
    if split == 'test':
        return load(dataset, split, data_dir=data_dir, emb_dir=f'{RELRAG}/data/cwq/emb_memmap/test')
    shard_dir = f'{ROOT}/data_files/cwq/emb/gte-large-en-v1.5'
    shard_index = json.load(open(f'{shard_dir}/shard_index.json'))
    return load(dataset, split, data_dir=data_dir, emb_dir=shard_dir,
                shard_index=shard_index, max_cache_shards=max_cache_shards)


def epoch_order(recs, emb):
    """Training-record order for one epoch. Plain shuffle for SubsetEmb (unchanged behavior).
    Shard-grouped shuffle for ShardEmbAdapter: shuffle shard order, then shuffle within each shard
    -- a flat shuffle over CWQ's 111 train shards with a 4-shard LRU cache would reload shards on
    almost every access (the ~9.6TB/epoch blowup ShardAwareSampler's docstring in
    src/dataset/retriever.py warns about)."""
    if not isinstance(emb, ShardEmbAdapter):
        order = recs[:]; random.shuffle(order); return order
    groups = defaultdict(list)
    for rec in recs:
        groups[emb.shard_key(rec['id'])].append(rec)
    keys = list(groups); random.shuffle(keys)
    order = []
    for k in keys:
        g = groups[k]; random.shuffle(g); order.extend(g)
    return order


def lexical(rec):
    """WL's lexical relation score (src/dataset/retriever.py), cached on the record"""
    if 'lex' not in rec:
        toks = clean_text(rec['question'])
        rec['lex'] = np.array([compute_lexical_score(toks, rel) for rel in rec['relation_list']], dtype=np.float32)
    return rec['lex']


def to_device(rec, emb, dev, need_lex=True):
    """need_lex: WL's lexical relation score (feat 'rt' only); False skips its ~10 ms of Python per question"""
    q, ent, rel = emb.get(rec['id'])
    topic = torch.zeros(rec['n_ent'], dtype=torch.long); topic[rec['q_ent']] = 1
    return {'lex': torch.from_numpy(lexical(rec)).to(dev) if need_lex else None, 'h': torch.from_numpy(rec['h'].astype(np.int64)).to(dev), 'r': torch.from_numpy(rec['r'].astype(np.int64)).to(dev),
            't': torch.from_numpy(rec['t'].astype(np.int64)).to(dev), 'q': q.float().to(dev), 'ent': ent.float().to(dev),
            'rel': rel.float().to(dev), 'topic': F.one_hot(topic, 2).float().to(dev),
            'label': torch.from_numpy(rec['label'].astype(np.float32)).to(dev), 'n_ent': rec['n_ent'],
            'rel_prior': torch.from_numpy(rec['rel_prior']).to(dev) if 'rel_prior' in rec else None,
            'a_ent': torch.tensor(rec.get('a_ent', []), dtype=torch.long, device=dev)}


class QRDDE(nn.Module):
    def __init__(self, emb=1024, d=64, K=3, use_query=True):
        super().__init__()
        self.u = nn.Linear(emb, d)
        self.use_query = use_query
        self.rel_mlp = nn.Sequential(nn.Linear((2 if use_query else 1) * emb, 256), nn.ReLU(), nn.Linear(256, 2 * d))
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(K))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(K))
        self.d = d

    def forward(self, x):
        n, d = x['n_ent'], self.d
        rel_in = torch.cat([x['rel'], x['q'].expand(len(x['rel']), -1)], 1) if self.use_query else x['rel']
        w_fwd, w_inv = self.rel_mlp(rel_in).split(d, dim=1)                          # per relation type
        h0 = x['topic'][:, 1:2] * self.u(x['q'])                                     # boundary condition
        src = torch.cat([x['h'], x['t']]); dst = torch.cat([x['t'], x['h']])
        w = torch.cat([w_fwd[x['r']], w_inv[x['r']]])
        deg = torch.zeros(n, device=src.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float)).clamp(min=1)
        h = h0
        for lin, ln in zip(self.layers, self.norms):
            agg = torch.zeros(n, d, device=src.device).index_add_(0, dst, h[src] * w) / deg[:, None]
            h = h + F.relu(ln(lin(h0 + agg)))
        return h


class Scorer(nn.Module):
    def __init__(self, feat, emb=1024, d=64, K=3, rtdde_floor=0.1, dde_rounds=2, hidden=None, layers=2, rbf_T=3, rbf_prior=True, rbf_pi='mlp', rbf_node=False, rbf_nb=False, rbf_mm=False, rel_prior='none', rt_src='gte'):
        """dde_rounds: forward = reverse DDE rounds (SubgraphRAG: 2). hidden / layers: MLP width (default = emb, as in
        SubgraphRAG) and depth (2 = Linear-ReLU-Linear, as in SubgraphRAG). Defaults reproduce the original module tree,
        so old checkpoints load unchanged."""
        super().__init__()
        self.feat, self.rt_src = feat, rt_src
        tokens = feat.split('+')
        if 'rbfonly' in tokens:          # RBF-Ret alone (rbf.py): no SubgraphRAG MLP, the path score is the logit
            self.rbf = RBF(emb, T=rbf_T, prior=rbf_prior, pi=rbf_pi, node=rbf_node, nb=rbf_nb, mm=rbf_mm, extra_prior=REL_PRIOR_DIM[rel_prior]); return
        if 'rbfpoe' in tokens:
            # product of experts: P(evidence) = P(on a reliable path from an anchor; RBF) x P(locally relevant; MLP on the
            # triple's text only: q, z_h, z_r, z_t -- no anchor flags, no DDE, no path features, so it cannot encode a
            # composition); trained jointly with one BCE on the product
            self.rbf = RBF(emb, T=rbf_T, prior=rbf_prior, pi=rbf_pi, node=rbf_node, nb=rbf_nb, mm=rbf_mm, extra_prior=REL_PRIOR_DIM[rel_prior])
            self.non_text = nn.Embedding(1, emb)
            hidden = hidden or emb
            self.loc = nn.Sequential(nn.Linear(4 * emb, hidden), nn.ReLU(), nn.Linear(hidden, 1)); return
        self.non_text = nn.Embedding(1, emb)
        ent_dim = emb + 2
        if 'dde' in tokens:
            self.dde = DDE(num_rounds=dde_rounds, num_reverse_rounds=dde_rounds); ent_dim += 2 * 2 * dde_rounds
        if 'rtdde' in tokens:
            self.rtdde = DDE(num_rounds=2, num_reverse_rounds=2); ent_dim += 2 * 4
            self.rtdde_floor = rtdde_floor
        if 'ase' in tokens:
            # anchor-set encoding (EXP-31, V1): DDE computed separately for every anchor, embedded by a shared phi and
            # pooled over the anchor set with permutation-invariant mean / max / min (union / intersection-like)
            self.ase_dde = DDE(num_rounds=dde_rounds, num_reverse_rounds=dde_rounds)
            k_in = 2 + 2 * 2 * dde_rounds
            self.ase_phi = nn.Sequential(nn.Linear(k_in, d), nn.ReLU(), nn.Linear(d, d))
            ent_dim += 3 * d
        if 'qrdde' in tokens or 'qrdde_noq' in tokens:
            self.qr = QRDDE(emb, d, K, use_query=('qrdde_noq' not in tokens)); ent_dim += d
        trip_extra = 0
        if 'rbf' in tokens:
            # RBF-Ret (rbf.py) path scores as 4 extra triple features of the SubgraphRAG MLP (+ auxiliary BCE on the
            # path score, --rbf_aux)
            self.rbf = RBF(emb, T=rbf_T, prior=rbf_prior, pi=rbf_pi, node=rbf_node, nb=rbf_nb, mm=rbf_mm, extra_prior=REL_PRIOR_DIM[rel_prior]); trip_extra += 4
        if 'rt' in tokens:
            self.rt_dde = DDE(num_rounds=dde_rounds, num_reverse_rounds=dde_rounds); ent_dim += 2 * 2 * dde_rounds
            trip_extra += 2
        hidden = hidden or emb
        mlp = [nn.Linear(2 * emb + 2 * ent_dim + trip_extra, hidden), nn.ReLU()]
        for _ in range(layers - 2):
            mlp += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.pred = nn.Sequential(*mlp, nn.Linear(hidden, 1))

    def forward(self, x):
        if 'rbfpoe' in self.feat.split('+'):
            o = self.rbf(x)
            path_lp = o['final'] - 0.6931471805599453                       # log P(path); = union for one anchor
            nt = self.non_text.weight.expand(x['n_ent'] - len(x['ent']), -1)
            he = torch.cat([x['ent'], nt])
            loc = self.loc(torch.cat([x['q'].expand(len(x['r']), -1), he[x['h']], x['rel'][x['r']], he[x['t']]], 1)).view(-1)
            self.rbf_logit, self.loc_logit = log_to_logit(path_lp), loc
            return log_to_logit(path_lp + F.logsigmoid(loc))
        if 'rbfonly' in self.feat.split('+'):
            o = self.rbf(x); self.rbf_out = o
            self.rbf_logit = log_to_logit(o['final'] - 0.6931471805599453)   # = union for one anchor
            return self.rbf_logit
        n_nt = x['n_ent'] - len(x['ent'])
        nt = self.non_text.weight.expand(n_nt, -1)
        if 'cvtrel' in self.feat.split('+') and n_nt > 0:
            # mediator (CVT) nodes have no text: encode each by the mean text embedding of its incident relations,
            # added to the shared learned vector -- relational identity without per-relation parameters
            n_text = len(x['ent']); dev_ = x['rel'].device
            acc = torch.zeros(x['n_ent'], x['rel'].shape[1], device=dev_)
            cnt = torch.zeros(x['n_ent'], device=dev_)
            ends = torch.cat([x['h'], x['t']]); rr = torch.cat([x['r'], x['r']])
            acc.index_add_(0, ends, x['rel'][rr]); cnt.index_add_(0, ends, torch.ones_like(ends, dtype=torch.float))
            nt = nt + acc[n_text:] / cnt[n_text:].clamp(min=1)[:, None]
        he = [torch.cat([x['ent'], nt]), x['topic']]
        if hasattr(self, 'dde'):
            ei = torch.stack([x['h'], x['t']]); he += self.dde(x['topic'], ei, ei.flip(0))
        if hasattr(self, 'rtdde'):
            ei = torch.stack([x['h'], x['t']])
            sem = (F.normalize(x['rel'], dim=-1) @ F.normalize(x['q'], dim=-1).T).squeeze(-1)  # cos(z_r, z_q) per relation type
            ew = torch.clamp(sem[x['r']], min=self.rtdde_floor)        # static, non-learned edge weight per triple
            he += self.rtdde(x['topic'], ei, ei.flip(0), edge_weight=ew)
        if hasattr(self, 'ase_dde'):
            ei = torch.stack([x['h'], x['t']]); n = x['n_ent']
            anchors = x['topic'][:, 1].nonzero().reshape(-1)
            per = []
            for a in anchors.tolist():                                   # one DDE per anchor
                oh = torch.zeros(n, 2, device=ei.device); oh[:, 0] = 1; oh[a, 0] = 0; oh[a, 1] = 1
                per.append(self.ase_phi(torch.cat([oh] + self.ase_dde(oh, ei, ei.flip(0)), 1)))
            P = torch.stack(per) if per else torch.zeros(1, n, self.ase_phi[-1].out_features, device=ei.device)
            he.append(torch.cat([P.mean(0), P.max(0).values, P.min(0).values], 1))
        if hasattr(self, 'qr'):
            he.append(self.qr(x))
        trip = []
        if hasattr(self, 'rt_dde'):
            ei = torch.stack([x['h'], x['t']])
            sem = x['rel_prior'][:, 0] if self.rt_src == 'bge' else (x['rel'] @ x['q'].T).squeeze(-1)
            s = 0.5 * x['lex'] + 0.5 * sem                                         # hybrid, alpha 0.5 (WL)
            mask = torch.zeros_like(s); mask[torch.topk(s, min(5, len(s))).indices] = 1
            he += self.rt_dde(x['topic'], ei, ei.flip(0), edge_weight=torch.clamp(s[x['r']], min=0.1))
            trip.append(F.one_hot(mask[x['r']].long(), 2).float())
        if hasattr(self, 'rbf'):
            o = self.rbf(x)
            self.rbf_logit = log_to_logit(o['final'] - 0.6931471805599453)
            rl = o['rel_logit'][x['r']]
            trip.append(torch.stack([o['union'].clamp(min=-20), o['inter'].clamp(min=-20), rl[:, 0], rl[:, 1]], 1) / 10)
        he = torch.cat(he, 1)
        feats = torch.cat([x['q'].expand(len(x['r']), -1), he[x['h']], x['rel'][x['r']], he[x['t']]] + trip, 1)
        return self.pred(feats).view(-1)


def build_scorer(a):
    """Scorer from a result.json 'args' dict (older runs lack the architecture keys -> SubgraphRAG defaults)."""
    return Scorer(a['feat'], d=a['d'], K=a['K'], dde_rounds=a.get('dde_rounds', 2), hidden=a.get('hidden'),
                  layers=a.get('layers', 2), rbf_T=a.get('rbf_T', 3), rbf_prior=bool(a.get('rbf_prior', 1)), rbf_pi=a.get('rbf_pi', 'mlp'), rbf_node=bool(a.get('rbf_node', 0)), rbf_nb=bool(a.get('rbf_nb', 0)), rbf_mm=bool(a.get('rbf_mm', 0)), rel_prior=a.get('rel_prior', 'none'), rt_src=a.get('rt_src', 'gte'))


def pseudo_anchor(rec):
    """a text entity adjacent to a gold answer that is not a topic and not an endpoint of a labelled triple: plays the
    role of a second constraint anchor (conjunction) without changing the labels"""
    lab = rec['label'].astype(bool); n_text = rec.get('n_text', rec['n_ent'])
    on_label = set(rec['h'][lab].tolist()) | set(rec['t'][lab].tolist())
    ans = set(rec['a_ent']); tops = set(rec['q_ent'])
    m = np.isin(rec['h'], list(ans)) | np.isin(rec['t'], list(ans))
    cand = [e for e in set(rec['h'][m].tolist()) | set(rec['t'][m].tolist())
            if e not in ans and e not in tops and e not in on_label and e < n_text]
    return random.choice(cand) if cand else None


def with_anchors(rec, anchors):
    return {**rec, 'q_ent': sorted(anchors)}


@torch.no_grad()
def bridges(rec, logits, B):
    """entities of the top-B triples other than the current anchors"""
    top = torch.topk(logits, min(B, len(logits))).indices.cpu().numpy()
    out = set(rec['h'][top].tolist()) | set(rec['t'][top].tolist())
    return out - set(rec['q_ent'])


def score_passes(model, rec, emb, dev, passes=1, B=3):
    """logits of a single pass, or the max of z-normalised logits over passes with re-anchoring (AER)"""
    x = to_device(rec, emb, dev)
    s = model(x)
    if passes == 1:
        return s
    z = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    best, anchors = z(s), set(rec['q_ent'])
    for _ in range(passes - 1):
        new = bridges({**rec, 'q_ent': sorted(anchors)}, s, B)
        if not new:
            break
        anchors |= new
        s = model(to_device(with_anchors(rec, anchors), emb, dev))
        best = torch.maximum(best, z(s))
    return best


@torch.no_grad()
def evaluate(model, recs, emb, dev, export_k=0, passes=1, B=3):
    model.eval()
    m = defaultdict(list); exported = {}; times = []
    for rec in recs:
        sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
        sync(); t0 = time.perf_counter()
        logits = score_passes(model, rec, emb, dev, passes, B)
        sync(); times.append(time.perf_counter() - t0)
        order = torch.argsort(logits, descending=True).cpu().numpy()
        grp = 'multi' if len(rec['q_ent']) > 1 else 'single'
        lab = rec['label'].astype(bool)
        for k in K_LIST:
            top = order[:k]
            ents = set(rec['h'][top].tolist()) | set(rec['t'][top].tolist())
            ar = len(set(rec['a_ent']) & ents) / len(rec['a_ent'])
            m[f'ans_recall@{k}'].append(ar); m[f'{grp}/ans_recall@{k}'].append(ar)
            if lab.any():
                m[f'triple_recall@{k}'].append(lab[top].sum() / lab.sum())
        if export_k:
            n, rl = rec['entity_names'], rec['relation_list']
            top = order[:export_k]; sc = logits.cpu().numpy()
            exported[rec['id']] = {'question': rec['question'], 'scored_triples': [
                (n[rec['h'][i]], rl[rec['r'][i]], n[rec['t'][i]], float(sc[i])) for i in top]}
    out = {k: float(np.mean(v)) for k, v in m.items()}
    out['n_multi'] = len(m['multi/ans_recall@20']); out['ms_per_query'] = 1000 * float(np.median(times))
    model.train()
    return out, exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--feat', required=True)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--dde_rounds', type=int, default=2, help='forward = reverse DDE rounds (SubgraphRAG: 2)')
    ap.add_argument('--hidden', type=int, default=None, help='MLP hidden width (default: embedding size, as SubgraphRAG)')
    ap.add_argument('--layers', type=int, default=2, help='MLP depth in Linear layers (SubgraphRAG: 2)')
    # Defaults = retrieve/configs/retriever/{webqsp,cwq}.yaml (num_epochs 10000, patience 10, seed 42).
    ap.add_argument('--epochs', type=int, default=10000)
    ap.add_argument('--patience', type=int, default=10, help='stop if val triple_recall@100 has not improved for this many epochs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--full', action='store_true', help='use exp_qrdde/data/ + the real WebQSP/CWQ embedding stores (M2) instead of the 800/1500-question subset')
    ap.add_argument('--max_cache_shards', type=int, default=4)
    ap.add_argument('--resume', default=None, help='continue training from these weights (a best.pt); only weights '
                    'were saved, so Adam restarts from zero moments')
    ap.add_argument('--start_epoch', type=int, default=0, help='epoch number of the first resumed epoch')
    ap.add_argument('--infer_passes', type=int, default=1, help='AER: 2 = re-anchor at bridges and fuse by max z-score (val, test, export)')
    ap.add_argument('--reanchor_B', type=int, default=3, help='AER: bridges = entities of the top-B triples')
    ap.add_argument('--reanchor_p', type=float, default=0.0, help='AER: probability of training a question on its pass-2 anchor set')
    ap.add_argument('--pseudo_anchor_p', type=float, default=0.0, help='EXP-31 V2: prob. of adding a pseudo constraint anchor '
                    '(a text entity adjacent to a gold answer, not on the labelled triples) so single-topic WebQSP questions look multi-anchor')
    ap.add_argument('--device', default='cuda:0', help="'cpu' is enough for --feat rbfonly (~15 ms per question)")
    ap.add_argument('--rbf_T', type=int, default=3, help='RBF-Ret propagation steps (training and default inference)')
    ap.add_argument('--rbf_prior', type=int, default=1, help='1 = feed the non-learned WL relation prior to pi(r|q)')
    ap.add_argument('--rbf_pi', default='mlp', choices=['mlp', 'gate'], help="RBF relation operator: 'gate' = sigmoid(tau . WL prior + b)")
    ap.add_argument('--rbf_node', type=int, default=0, help='1 = RBF node factor psi(v | q) (rbf.py)')
    ap.add_argument('--rbf_nb', type=int, default=0, help='1 = non-backtracking walks (rbf.py)')
    ap.add_argument('--rbf_mm', type=int, default=0, help='1 = complete-path max-marginal scoring (rbf.py)')
    ap.add_argument('--rel_prior', default='none', choices=list(REL_PRIOR_DIM), help="extra relation prior for pi: 'pmh' = "
                    "Path+Memory Hybrid selector cache (in-distribution only)")
    ap.add_argument('--rt_src', default='gte', choices=['gte', 'bge'], help="feat 'rt': semantic half of the relation score = "
                    "GTE <z_r, z_q> (WL) or the fine-tuned BGE cosine of --rel_prior bge/bge2 (WL + BGE, Phase 11)")
    ap.add_argument('--rbf_ans_loss', type=float, default=0.0, help='weight of the listwise answer loss on RBF best-path '
                    'scores (answer_softmax_loss); 0 = BCE only')
    ap.add_argument('--rbf_aux', type=float, default=0.5, help="feat 'rbf': weight of the BCE on the RBF path-score logit")
    ap.add_argument('--label_src', default='sp', choices=['sp', 'rational', 'union', 'sparql', 'sparql_chain', 'sparql2', 'sparql3'], help="training labels: SubgraphRAG shortest "
                    "paths, the hybrid GPT-4o + DeepSeek rational paths (make_rational.py; SP where a question has none), or "
                    "their union (35%% of rational triples lie on no shortest path). "
                    "Model selection stays on SP triple recall")
    ap.add_argument('--pos_weight', default='none', choices=['none', 'effective_num'], help="'effective_num' = WL's loss")
    ap.add_argument('--anchor_aug_p', type=float, default=0.0, help='EXP-35 matched anchor augmentation probability')
    ap.add_argument('--sub_dir', default=None, help='train/val from <sub_dir>/{train,val}.pkl + emb_{split} (SubsetEmb layout), e.g. the CWQ subset of EXP-31')
    ap.add_argument('--lean_emb', type=int, default=0, help='1 = train/val from data/<ds>/lemb_<split> (make_lean_emb.py): '
                    'question + relation vectors and precomputed lex only, no entity vectors (RBF-only)')
    ap.add_argument('--skip_test', action='store_true', help='no test eval/export at the end (scored later, e.g. with adr_probe.py)')
    args = ap.parse_args()
    set_seed(args.seed, deterministic=True)   # same as retrieve/train.py; must run before any CUDA call
    dev = torch.device(args.device); os.makedirs(args.out, exist_ok=True)
    if args.sub_dir:
        train, emb_tr = load(args.dataset, 'train', data_dir=args.sub_dir)
        val, emb_va = load(args.dataset, 'val', data_dir=args.sub_dir)
    else:
        if args.lean_emb:
            assert args.feat == 'rbfonly' and not args.rbf_node, '--lean_emb drops entity vectors: RBF-only, no psi'
        train, emb_tr = resolve(args.dataset, 'train', args.full, args.max_cache_shards, lean=bool(args.lean_emb))
        val, emb_va = resolve(args.dataset, 'val', args.full, args.max_cache_shards, lean=bool(args.lean_emb))
    if not args.skip_test:
        test, emb_te = resolve(args.dataset, 'test', args.full, args.max_cache_shards)
        attach_rel_prior(test, args.dataset, 'test', args.rel_prior, args.dataset)
    if args.label_src.startswith('sparql'):
        for recs_, sp_ in ((train, 'train'), (val, 'val')):
            apply_sparql_labels(recs_, args.dataset, sp_, with_constraints=args.label_src != 'sparql_chain',
                                version={'sparql2': 'v2', 'sparql3': 'v3'}.get(args.label_src, ''))
    attach_rel_prior(train, args.dataset, 'train', args.rel_prior, args.dataset)
    attach_rel_prior(val, args.dataset, 'val', args.rel_prior, args.dataset)
    model = build_scorer(vars(args)).to(dev)
    rational = pickle.load(open(f'{ROOT}/exp_qrdde/data/{args.dataset}/rational_train.pkl', 'rb')) if args.label_src in ('rational', 'union') else {}
    aug = pickle.load(open(f'{ROOT}/exp_qrdde/data/{args.dataset}/anchor_aug.pkl', 'rb')) if args.anchor_aug_p > 0 else {}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Same wandb logging as retrieve/train.py (train loss + patience every epoch, val metrics only on
    # improvement), under one shared project. Set WANDB_MODE=disabled to turn it off.
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    wandb.init(project='REL-RAG', name=f'{os.path.basename(os.path.normpath(args.out))}_{ts}', config=vars(args))
    best, since_best, log = -1, 0, []
    if args.resume:
        # the resumed weights are the current best: score them on val so patience counts from them
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        v, _ = evaluate(model, val, emb_va, dev, passes=args.infer_passes, B=args.reanchor_B)
        best = v['triple_recall@100']; torch.save(model.state_dict(), f'{args.out}/best.pt')
        log.append({'epoch': args.start_epoch - 1, 'resumed_from': args.resume, **{f'val_{k}': x for k, x in v.items()}})
        wandb.log({'val/epoch': args.start_epoch - 1, **{f'val/{k}': x for k, x in v.items()}})
        print(json.dumps({'resumed_from': args.resume, 'val': {k: round(x, 4) for k, x in v.items()}}), flush=True)
    for ep in range(args.start_epoch, args.epochs):
        order = epoch_order(train, emb_tr); t0 = time.time(); tot = 0
        for rec in tqdm(order):
            x = to_device(rec, emb_tr, dev)
            if args.reanchor_p > 0 and random.random() < args.reanchor_p:
                with torch.no_grad():
                    model.eval(); new = bridges(rec, model(x), args.reanchor_B); model.train()
                if new:
                    x = to_device(with_anchors(rec, set(rec['q_ent']) | new), emb_tr, dev)
            if args.pseudo_anchor_p > 0 and random.random() < args.pseudo_anchor_p:
                pa = pseudo_anchor(rec)
                if pa is not None:
                    x = to_device(with_anchors(rec, set(x['topic'][:, 1].nonzero().reshape(-1).tolist()) | {pa}), emb_tr, dev)
            if args.anchor_aug_p > 0 and rec['id'] in aug and random.random() < args.anchor_aug_p:
                c, lab = random.choice(aug[rec['id']])
                x = to_device(with_anchors(rec, set(rec['q_ent']) | {c}), emb_tr, dev)
                x['label'][torch.tensor(lab, device=dev)] = 1.0
            if rec['id'] in rational:
                if args.label_src == 'rational':
                    x['label'] = torch.zeros_like(x['label'])
                x['label'][torch.tensor(rational[rec['id']], device=dev)] = 1.0
            pw = None
            if args.pos_weight == 'effective_num':
                n_pos = int(x['label'].sum().item()); n_neg = len(x['label']) - n_pos
                if n_pos > 0:
                    eff = lambda n: (1.0 - 0.999 ** n) / (1.0 - 0.999)
                    pw = torch.tensor(eff(n_neg) / (eff(n_pos) + 1e-8), device=dev)
            loss = F.binary_cross_entropy_with_logits(model(x), x['label'], pos_weight=pw)
            if args.rbf_ans_loss > 0 and hasattr(model, 'rbf_out'):
                loss = loss + args.rbf_ans_loss * answer_softmax_loss(model.rbf_out, x)
            if 'rbf' in args.feat.split('+') and args.rbf_aux > 0:
                loss = loss + args.rbf_aux * F.binary_cross_entropy_with_logits(model.rbf_logit, x['label'], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        v, _ = evaluate(model, val, emb_va, dev, passes=args.infer_passes, B=args.reanchor_B)
        log.append({'epoch': ep, 'loss': tot / len(train), 'sec': time.time() - t0, **{f'val_{k}': x for k, x in v.items()}})
        if v['triple_recall@100'] > best:
            best = v['triple_recall@100']; since_best = 0; torch.save(model.state_dict(), f'{args.out}/best.pt')
            wandb.log({'val/epoch': ep, **{f'val/{k}': x for k, x in v.items()}})
        else:
            since_best += 1
        wandb.log({'loss': tot / len(train), 'num_patient_epochs': since_best, 'epoch': ep})
        if args.patience is not None and since_best >= args.patience:   # same stop rule as retrieve/train.py
            break
    model.load_state_dict(torch.load(f'{args.out}/best.pt'))
    if args.skip_test:
        best_ep = max(log, key=lambda r: r['val_triple_recall@100'])['epoch']
        json.dump({'args': vars(args), 'test': {'best_epoch': best_ep}, 'log': log}, open(f'{args.out}/result.json', 'w'), indent=1)
        print(json.dumps({'run': args.out, 'best_epoch': best_ep, 'skip_test': True})); return
    res, exported = evaluate(model, test, emb_te, dev, export_k=200, passes=args.infer_passes, B=args.reanchor_B)
    res['params'] = sum(p.numel() for p in model.parameters())
    res['best_epoch'] = max(log, key=lambda r: r['val_triple_recall@100'])['epoch']
    json.dump({'args': vars(args), 'test': res, 'log': log}, open(f'{args.out}/result.json', 'w'), indent=1)
    torch.save(exported, f'{args.out}/retrieval_test.pth')
    print(json.dumps({'run': args.out, **{k: round(v, 4) for k, v in res.items()}}))


if __name__ == '__main__':
    main()

