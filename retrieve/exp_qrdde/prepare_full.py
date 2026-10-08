"""Full (non-sampled) WebQSP / CWQ records for the M2 full-data QR-DDE experiments
(docs/01_RESEARCH_PLAN.md). Same compact record schema and eligibility filter as
prepare_subsets.py, minus the rng.sample -- every eligible question is kept. Unlike
prepare_subsets.py, embeddings are never copied into a new memmap here:
  - WebQSP (all splits) and CWQ test reuse the existing memmaps directly
    (/relrag/data/emb_memmap/<split>, /relrag/data/cwq/emb_memmap/test) via qrdde.py's
    resolve(..., full=True) -- verified id-for-id identical to the SubsetEmb layout.
  - CWQ train/val are read straight from the GTE shard files at train time (ShardEmbAdapter);
    materializing a memmap for them would need ~43G and only ~15-27G disk is free.
CWQ train.pkl is 1.7G and needs ~14GB RAM to unpickle -- run `--dataset cwq --split train` alone
inside its own memory-capped systemd-run invocation (see run_full.sh), never concurrently with
another heavy job.
"""
import argparse, json, os, pickle, sys
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG/retrieve'
RELRAG = '/home/minhnv/Lab/SubgraphRAG/relrag'   # REL-RAG reproduction (moved out of RAPL/ on 10-07)
OUT = f'{ROOT}/exp_qrdde/data'
sys.path.insert(0, ROOT)


def _record(s, ts_entry):
    return {
        'id': s['id'], 'question': s['question'],
        'h': np.array(s['h_id_list'], dtype=np.int32), 'r': np.array(s['r_id_list'], dtype=np.int32),
        't': np.array(s['t_id_list'], dtype=np.int32),
        'q_ent': sorted(set(s['q_entity_id_list'])), 'a_ent': sorted(set(s['a_entity_id_list'])),
        'n_text': len(s['text_entity_list']), 'n_ent': len(s['text_entity_list']) + len(s['non_text_entity_list']),
        'label': ts_entry['triple_scores'].numpy().astype(np.float16),
        'entity_names': list(s['text_entity_list']) + list(s['non_text_entity_list']),
        'relation_list': list(s['relation_list']),
    }


def build_webqsp(split):
    out_dir = f'{OUT}/webqsp'; os.makedirs(out_dir, exist_ok=True)
    outp = f'{out_dir}/{split}.pkl'
    if os.path.exists(outp):
        print('webqsp', split, 'exists'); return
    sys.path.insert(0, f'{RELRAG}')
    from dataset import MemmapEmbeddings
    emb = MemmapEmbeddings(split)
    raw = pickle.load(open(f'{ROOT}/data_files/webqsp/processed/{split}.pkl', 'rb'))
    ts = torch.load(f'{ROOT}/data_files/webqsp/triple_scores/{split}.pth', weights_only=False)
    ok = [s for s in raw if s['id'] in emb and len(s['h_id_list']) > 0 and s['a_entity_id_list']
          and s['q_entity_id_list'] and (split == 'test' or ts[s['id']]['triple_scores'].sum() > 0)]
    recs = [_record(s, ts[s['id']]) for s in ok]
    del raw
    pickle.dump(recs, open(outp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('webqsp', split, len(recs), 'records (eligible', len(ok), ')', flush=True)


def build_cwq(split, shard_index=None):
    """CWQ train's raw processed.pkl unpickles to ~12-13GB; building the full compact record list
    and then pickle.dump()-ing it as one object drove peak RSS past 26GB (glibc heap fragmentation
    across ~22k small numpy-array records never gets returned to the OS between those two steps).
    Instead: stream one pickle.dump() per record straight from the `raw` iteration (a fresh Pickler
    per call, so no cross-record memo growth) to a temp file -- peak stays pinned near raw's own
    ~13GB. A second, cheap pass then re-reads that (much smaller, ~1-2GB) stream, sorts it for
    shard locality, and re-dumps it as the single list object qrdde.py's load() expects."""
    out_dir = f'{OUT}/cwq'; os.makedirs(out_dir, exist_ok=True)
    outp = f'{out_dir}/{split}.pkl'
    if os.path.exists(outp):
        print('cwq', split, 'exists'); return
    tmp = f'{outp}.stream_tmp'
    raw = pickle.load(open(f'{ROOT}/data_files/cwq/processed/{split}.pkl', 'rb'))
    ts = torch.load(f'{ROOT}/data_files/cwq/triple_scores/{split}.pth', weights_only=False)
    if split == 'test':
        sys.path.insert(0, f'{RELRAG}')
        from dataset import MemmapEmbeddings
        emb = MemmapEmbeddings('test', dataset='cwq')
        has_emb = lambda sid: sid in emb
    else:
        assert shard_index is not None
        has_emb = lambda sid: str(sid) in shard_index
    n = 0
    with open(tmp, 'wb') as f:
        for s in raw:
            if not (has_emb(s['id']) and len(s['h_id_list']) > 0 and s['a_entity_id_list']
                    and s['q_entity_id_list'] and (split == 'test' or ts[s['id']]['triple_scores'].sum() > 0)):
                continue
            pickle.dump(_record(s, ts[s['id']]), f, protocol=pickle.HIGHEST_PROTOCOL)
            n += 1
    del raw, ts

    recs = []
    with open(tmp, 'rb') as f:
        while True:
            try:
                recs.append(pickle.load(f))
            except EOFError:
                break
    os.remove(tmp)
    if split in ('train', 'val'):
        recs.sort(key=lambda r: shard_index[str(r['id'])])  # shard-local order, avoids thrashing later
    pickle.dump(recs, open(outp, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('cwq', split, len(recs), 'records (eligible', n, ')', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'])
    ap.add_argument('--split', choices=['train', 'val', 'test'])
    args = ap.parse_args()
    splits = [args.split] if args.split else ['train', 'val', 'test']
    if args.dataset == 'webqsp':
        for sp in splits:
            build_webqsp(sp)
        return
    shard_index = None
    if any(sp in ('train', 'val') for sp in splits):
        d = f'{ROOT}/data_files/cwq/emb/gte-large-en-v1.5'
        shard_index = json.load(open(f'{d}/shard_index.json'))
    for sp in splits:
        build_cwq(sp, shard_index)


if __name__ == '__main__':
    main()
