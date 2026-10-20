"""Line-graph samples for REL-RAG, built on the fly from the compact records of prepare_data.py.

Node i of the line graph is triple (h_i, r_i, t_i); a directed edge i -> j exists iff t_i == h_j
(paper Def. 4). Node features follow the original RAPL code: [h_emb | r_emb | t_emb] (3072-d),
with zero vectors for non-text (CVT) entities. The full line graph is used (no edge cap).
"""
import json, os, random, pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = os.environ.get('RELRAG_ROOT', '/home/minhnv/Lab/SubgraphRAG')
DATA_DIR = f'{ROOT}/relrag/data'
EMB_DIR = f'{ROOT}/retrieve/data_files/webqsp/emb/gte-large-en-v1.5'
MEMMAP_DIR = f'{DATA_DIR}/emb_memmap'


class MemmapEmbeddings:
    """Read-only float16 embeddings backed by np.memmap (see make_memmap.py).

    Pages live in the reclaimable page cache, so RAM is not pinned per process; this replaced the
    in-RAM dict that crashed the host twice when combined with DataLoader workers and ollama.
    """
    def __init__(self, split, dataset='webqsp'):
        d = f'{MEMMAP_DIR}/{split}' if dataset == 'webqsp' else f'{DATA_DIR}/{dataset}/emb_memmap/{split}'
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))
        self.q = np.load(f'{d}/q.npy', mmap_mode='r')
        self.ent = np.load(f'{d}/ent.npy', mmap_mode='r')
        self.rel = np.load(f'{d}/rel.npy', mmap_mode='r')

    def __contains__(self, sid):
        return sid in self.index

    def __getitem__(self, sid):
        k, e0, ne, r0, nr = self.index[sid]
        return {'q_emb': torch.from_numpy(np.array(self.q[k:k + 1])),
                'entity_embs': torch.from_numpy(np.array(self.ent[e0:e0 + ne])),
                'relation_embs': torch.from_numpy(np.array(self.rel[r0:r0 + nr]))}


def load_embeddings(split, dataset='webqsp'):
    return MemmapEmbeddings(split, dataset)


def line_graph_edges(h, t):
    """Edges (src, dst) of the directed line graph: tail of src == head of dst."""
    n = len(h)
    num_ent = int(max(h.max(), t.max())) + 1
    tail_inc = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), t)), shape=(n, num_ent))
    head_inc = sp.csr_matrix((np.ones(n, dtype=np.float32), (h, np.arange(n))), shape=(num_ent, n))
    lg = (tail_inc @ head_inc).tocsr()   # lg[i, j] = 1 iff t_i == h_j
    lg.sort_indices()
    return lg


def load_label_paths(recs, label_source, tau, emb):
    """Attach `paths` (training label paths) and `topic_pos` to each record.

    label_source: 'sp'  -> all forward shortest paths (paper's ShortestPath labels)
                  'paths:<file>' -> rational-path labels from merge_labels.py: the authors' d..d+2 candidate
                                    instances whose relation path the LLM chose (App. B), as line-graph node ids
    topic_pos implements paper Eq. 10: first triples of label paths plus question triples whose
    relation embedding has cosine similarity > tau with a ground-truth first relation r*.
    """
    given = None
    if label_source.startswith('paths:'):
        given = {d['id']: d['paths'] for d in map(json.loads, open(label_source[6:]))}
    elif label_source != 'sp':
        raise ValueError(f'unknown label source {label_source}')
    stats = {'samples': 0, 'with_label': 0, 'llm_fallback_sp': 0}
    for rec in recs:
        stats['samples'] += 1
        paths = rec['sp_paths']
        if given is not None:
            if given.get(rec['id']):
                paths = given[rec['id']]
            else:
                # no annotation / LLM rejected every candidate: keep shortest paths so the sample is not lost
                stats['llm_fallback_sp'] += 1
        rec['paths'] = paths
        stats['with_label'] += bool(paths)

        pos = {p[0] for p in paths}
        if paths and len(rec['topic_candidates']):
            rel = F.normalize(emb[rec['id']]['relation_embs'].float(), dim=-1)
            rstar = torch.tensor(sorted({int(rec['r'][p[0]]) for p in paths}))
            cand = torch.from_numpy(rec['topic_candidates'].astype(np.int64))
            cand_r = torch.from_numpy(rec['r'][rec['topic_candidates']].astype(np.int64))
            sim = (rel[cand_r] @ rel[rstar].T).max(dim=1).values
            pos.update(cand[sim > tau].tolist())
        rec['topic_pos'] = np.array(sorted(pos), dtype=np.int64)
    return stats


class LineGraphSample:
    __slots__ = ('id', 'x', 'q', 'adj_fwd', 'nbr_ptr', 'nbr_idx', 'topic_cand', 'topic_pos',
                 'paths', 'h', 'r', 't', 'rec')


def build_sample(rec, emb):
    e = emb[rec['id']]
    h = rec['h'].astype(np.int64); r = rec['r'].astype(np.int64); t = rec['t'].astype(np.int64)
    ent = e['entity_embs']
    n_text = ent.shape[0]
    ent_full = torch.cat([ent, torch.zeros(1, ent.shape[1], dtype=ent.dtype)], 0)  # last row = zeros
    hi = torch.from_numpy(np.where(h < n_text, h, n_text))
    ti = torch.from_numpy(np.where(t < n_text, t, n_text))
    x = torch.cat([ent_full[hi], e['relation_embs'][torch.from_numpy(r)], ent_full[ti]], dim=1)

    lg = line_graph_edges(h, t)
    s = LineGraphSample()
    s.id = rec['id']; s.x = x; s.q = e['q_emb'].reshape(1, -1)
    coo = lg.tocoo()
    s.adj_fwd = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))  # src -> dst
    s.nbr_ptr = torch.from_numpy(lg.indptr.astype(np.int64))                     # out-neighbours
    s.nbr_idx = torch.from_numpy(lg.indices.astype(np.int64))
    s.topic_cand = torch.from_numpy(rec['topic_candidates'].astype(np.int64))
    s.topic_pos = torch.from_numpy(rec.get('topic_pos', np.zeros(0, dtype=np.int64)))
    s.paths = rec.get('paths', [])
    s.h, s.r, s.t = h, r, t
    s.rec = rec
    return s


class LineGraphDataset(torch.utils.data.Dataset):
    def __init__(self, split, emb, label_source=None, tau=0.95, require_label=False, dataset='webqsp'):
        recs = pickle.load(open(f'{DATA_DIR}/{split}.pkl' if dataset == 'webqsp' else f'{DATA_DIR}/{dataset}/{split}.pkl', 'rb'))
        recs = [r for r in recs if r['id'] in emb and len(r['h']) > 0]
        self.label_stats = None
        if label_source is not None:
            self.label_stats = load_label_paths(recs, label_source, tau, emb)
        if require_label:
            recs = [r for r in recs if r.get('paths')]
        self.recs, self.emb = recs, emb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return build_sample(self.recs[i], self.emb)


def collate_single(batch):
    return batch[0]
