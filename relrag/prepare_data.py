"""Stage 1 of the REL-RAG reproduction: compact per-sample records for the line-graph retriever.

For every sample we keep only integer arrays, so nothing large is written to disk:
  * line-graph nodes = edges of the entity DiGraph (one triple per (h, t) pair, as the
    original RAPL code does with nx.DiGraph), stored as h / r / t id arrays;
  * forward shortest paths from each question entity to each answer entity, as sequences
    of line-graph node indices (paper Sec. 4.1 / App. B "Acquiring training labels");
  * the unique relation paths of those shortest paths, used as candidates for LLM filtering.
Line-graph edges and node features are rebuilt on the fly by `dataset.py`.
"""
import os, pickle, argparse, itertools
from collections import defaultdict, deque
import numpy as np
from tqdm import tqdm

SUBGRAPHRAG_DIR = '/home/minhnv/Lab/SubgraphRAG/retrieve/data_files/webqsp'
OUT_DIR = '/home/minhnv/Lab/SubgraphRAG/relrag/data'

MAX_PATHS_PER_PAIR = 50     # guard against combinatorial explosion through CVT hubs
MAX_PATHS_PER_SAMPLE = 256


def bfs_dag(out_adj, src):
    """BFS distances from src plus predecessor lists of the shortest-path DAG."""
    dist = {src: 0}
    pred = defaultdict(list)
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in out_adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
            if dist[v] == dist[u] + 1:
                pred[v].append(u)
    return dist, pred


def shortest_paths(dist, pred, src, dst, limit):
    """All shortest entity paths src -> dst (walk back along the BFS DAG), capped at `limit`."""
    if dst not in dist:
        return []
    out, stack = [], [[dst]]
    while stack and len(out) < limit:
        p = stack.pop()
        u = p[-1]
        if u == src:
            out.append(p[::-1])
            continue
        for w in pred[u]:
            stack.append(p + [w])
    return out


def build_sample(s):
    h_all, r_all, t_all = s['h_id_list'], s['r_id_list'], s['t_id_list']
    # nx.DiGraph semantics: the last triple added for a (h, t) pair wins
    pair2trip = {}
    for i, (h, t) in enumerate(zip(h_all, t_all)):
        pair2trip[(h, t)] = i
    kept = sorted(pair2trip.values())
    h = np.array([h_all[i] for i in kept], dtype=np.int32)
    r = np.array([r_all[i] for i in kept], dtype=np.int32)
    t = np.array([t_all[i] for i in kept], dtype=np.int32)
    pair2node = {(int(h[k]), int(t[k])): k for k in range(len(kept))}

    out_adj = defaultdict(list)
    for (hh, tt) in pair2node:
        out_adj[hh].append(tt)

    q_ents = list(dict.fromkeys(s['q_entity_id_list']))
    a_ents = list(dict.fromkeys(s['a_entity_id_list']))
    paths = []  # list of lists of line-graph node indices
    for q in q_ents:
        dist, pred = bfs_dag(out_adj, q)
        for a in a_ents:
            if a == q:
                continue
            for ep in shortest_paths(dist, pred, q, a, MAX_PATHS_PER_PAIR):
                paths.append([pair2node[(ep[i], ep[i + 1])] for i in range(len(ep) - 1)])
    # dedupe, keep deterministic order, cap
    seen, uniq = set(), []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key); uniq.append(p)
    paths = uniq[:MAX_PATHS_PER_SAMPLE]

    q_set = set(q_ents)
    topic_candidates = np.array([k for k in range(len(kept)) if h[k] in q_set], dtype=np.int32)
    return {
        'id': s['id'], 'question': s['question'],
        'h': h, 'r': r, 't': t, 'orig_triple_idx': np.array(kept, dtype=np.int32),
        'q_entity_id_list': q_ents, 'a_entity_id_list': a_ents, 'a_entity': s['a_entity'],
        'topic_candidates': topic_candidates,
        'sp_paths': paths,
        'num_text_entities': len(s['text_entity_list']),
        'entity_names': list(s['text_entity_list']) + list(s['non_text_entity_list']),
        'relation_list': list(s['relation_list']),
    }


def relation_path_candidates(rec, max_unique=20):
    """Group shortest paths by relation sequence; one textual example per relation path."""
    groups = {}
    for pi, p in enumerate(rec['sp_paths']):
        key = tuple(rec['relation_list'][rec['r'][k]] for k in p)
        groups.setdefault(key, []).append(pi)
    items = sorted(groups.items(), key=lambda kv: (len(kv[0]), -len(kv[1])))[:max_unique]
    names = rec['entity_names']
    cands = []
    for rel_path, pis in items:
        p = rec['sp_paths'][pis[0]]
        text = names[rec['h'][p[0]]]
        for k in p:
            text += f" -> {rec['relation_list'][rec['r'][k]]} -> {names[rec['t'][k]]}"
        cands.append({'relations': list(rel_path), 'example': text, 'path_ids': pis})
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--dataset', default='webqsp')
    args = ap.parse_args()
    global SUBGRAPHRAG_DIR, OUT_DIR
    if args.dataset != 'webqsp':
        SUBGRAPHRAG_DIR = SUBGRAPHRAG_DIR.replace('/webqsp', f'/{args.dataset}')
        OUT_DIR = f'{OUT_DIR}/{args.dataset}'
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in args.splits.split(','):
        raw = pickle.load(open(f'{SUBGRAPHRAG_DIR}/processed/{split}.pkl', 'rb'))
        recs, n_paths, n_rel = [], [], []
        for s in tqdm(raw, desc=split):
            rec = build_sample(s)
            rec['rel_path_cands'] = relation_path_candidates(rec)
            recs.append(rec)
            n_paths.append(len(rec['sp_paths'])); n_rel.append(len(rec['rel_path_cands']))
        n_paths, n_rel = np.array(n_paths), np.array(n_rel)
        print(f'[{split}] samples={len(recs)} with_path={(n_paths > 0).sum()} '
              f'paths/sample mean={n_paths.mean():.1f} | unique rel-paths: '
              f'==1: {(n_rel == 1).sum()}  >1: {(n_rel > 1).sum()}  mean={n_rel[n_rel > 0].mean():.2f}')
        with open(f'{OUT_DIR}/{split}.pkl', 'wb') as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
