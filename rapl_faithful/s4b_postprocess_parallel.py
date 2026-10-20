"""Step 4b — the authors' post-processing (ProcessedDiskDataset._process_all) split over W processes. Same result:
_process_all handles each sample independently (_compute_and_add_new_attr(data_obj, idx) reads only sample idx's labels,
retrieval record and metadata, then writes data_<idx>.pt); its only shared state is min_len_indices_list, appended in
sample order, which is rebuilt here by concatenating the workers' lists in sample order. No randomness is involved.
The workers read the slim KGDataset files without rebuilding x (D12): post-processing never reads x or edge_index
(only num_nodes, stored on the slim object), and the saved file stays slim; x is rebuilt at inference time.
Self-check: for a few samples, the output is compared with the authors' sequential path (KGDataset.get -> full object).
Run from RAPL/ after the KGDataset build:   cd RAPL && python ../rapl_faithful/s4b_postprocess_parallel.py --workers 5
"""
import argparse, os, pickle, sys
from multiprocessing import Pool
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), '..', 'rapl_faithful'))
assert os.path.basename(os.getcwd()) == 'RAPL'
import torch
import slim_x
from src.dataset.PostProcessedDataset import ProcessedDiskDataset

name, split = 'cwq', 'test'
RAW, KG = f'data/{name}/raw', f'data/{name}/processed/{split}'
OUT = f'data/post_processed_pathLabel_4o_corrected/{name}/{split}'
LTR = f'data_files/{name}/topic_relation_candidates/{split}_results.pkl'
S = {}


def make_pp():
    """a ProcessedDiskDataset with the state its __init__ sets for first-time processing (no disk scan / build)"""
    pp = ProcessedDiskDataset.__new__(ProcessedDiskDataset)
    pp.split, pp.split_dir = split, OUT
    pp.retrieval_list, pp.text_labels, pp.metadata_list = S['retrieval_list'], S['text_labels'], S['metadata_list']
    pp.labeled_topic_relation = pp._load_labeled_topic_relation(LTR)
    pp.min_len_indices_list = []
    return pp


def work(idxs):
    pp = make_pp()
    for sample_idx in idxs:                                     # body of _process_all, verbatim
        data_obj = slim_x._real_load(os.path.join(KG, f'data_{sample_idx}.pt'))
        out_path = os.path.join(pp.split_dir, f'data_{sample_idx}.pt')
        try:
            data_obj = pp._compute_and_add_new_attr(data_obj, sample_idx)
            slim_x._real_save(data_obj, out_path)
        except Exception as e:
            print(f"Exception in sample_idx {sample_idx}: {e}", flush=True)
            slim_x._real_save(data_obj, out_path)
    return idxs[0], pp.min_len_indices_list


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    from src.dataset.PyG_dataset_disk import KGDataset
    S['retrieval_list'] = pickle.load(open(f'{RAW}/processed/{split}_retrieval.pkl', 'rb'))
    S['text_labels'] = KGDataset.__new__(KGDataset)._collect_annotated_paths(f'{RAW}/annotated_paths_GPT4o/{split}')
    S['metadata_list'] = pickle.load(open(f'{KG}/metadata_{split}.pkl', 'rb'))
    n = len(S['retrieval_list'])
    assert len(S['text_labels']) == len(S['metadata_list']) == n
    os.makedirs(OUT, exist_ok=True)
    size = (n + args.workers - 1) // args.workers
    chunks = [list(range(s, min(s + size, n))) for s in range(0, n, size)]
    with Pool(args.workers) as pool:
        res = sorted(pool.map(work, chunks))
    min_len = [m for _, part in res for m in part]
    assert len(min_len) == n
    with open(os.path.join(OUT, 'min_len_indices_list.pkl'), 'wb') as f:
        pickle.dump(min_len, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUT, 'metadata_list.pkl'), 'wb') as f:
        pickle.dump(S['metadata_list'], f, protocol=pickle.HIGHEST_PROTOCOL)
    # self-check against the authors' sequential path on a few samples (full object from KGDataset.get, x rebuilt)
    slim_x.install(f'{RAW}/gte-large-en-v1.5/test.pth', tag_build=False)
    pp = make_pp()
    for i in [0, 1, n // 2, n - 1]:
        ref = pp._compute_and_add_new_attr(torch.load(os.path.join(KG, f'data_{i}.pt')), i)   # full object
        got = torch.load(os.path.join(OUT, f'data_{i}.pt'))                                    # parallel output, x rebuilt
        for key in ref.keys():
            a, b = ref[key], got[key]
            ok = (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)) \
                or (not torch.is_tensor(a) and a == b)
            assert ok, f'sample {i}: attribute {key} differs from the sequential path'
    print(f'post-processed {n} samples with {args.workers} workers -> {OUT}; self-check OK on 4 samples')


if __name__ == '__main__':
    main()
