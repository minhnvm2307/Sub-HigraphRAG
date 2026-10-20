"""Step 4 — test line graphs (KGDataset) and the post-processed test set that RAPL/model_inference.py reads, built with
the authors' own classes. This is RAPL/load_post_processed_dataset.py line for line, with two differences forced by the
released code itself:
  - split 'test' (the released script loops only over ['train']; RAPL/load_post_processed_dataset_test.py also builds
    'val' first, which needs CWQ val inputs this evaluation does not use);
  - output dir data/post_processed_pathLabel_4o_corrected/<ds>/ (the dir model_inference.py loads; the released script
    writes data/post_processed_pathLabel/<ds>/).
Run from RAPL/:   cd RAPL && python ../rapl_faithful/s4_build_test_graphs.py
"""
import os, sys, pickle
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), '..', 'rapl_faithful'))
assert os.path.basename(os.getcwd()) == 'RAPL', 'run from the RAPL/ directory'
from src.dataset.PyG_dataset_disk import KGDataset
from src.dataset.PostProcessedDataset import ProcessedDiskDataset

import s4_build_test_graphs_patch  # noqa: F401  D11: whitespace fix in KGDataset._paths_to_triplets (see that file)
import slim_x                      # D12: line graphs saved without x, x rebuilt exactly at load (see slim_x.py)
slim_x.install('data/cwq/raw/gte-large-en-v1.5/test.pth')

name, split = 'cwq', 'test'
post_processed_dir = f'data/post_processed_pathLabel_4o_corrected/{name}/'
dset = KGDataset(root=f'data/{name}/', split=split)            # builds data/cwq/processed/test/data_*.pt + metadata
raw_dataset = dset
raw_dir = f'data/{name}/raw/'
processed_dir = f'data/{name}/processed'
with open(f'{raw_dir}/processed/{split}_retrieval.pkl', 'rb') as f:
    retrieval_list = pickle.load(f)
text_labels = dset._collect_annotated_paths(f'{raw_dir}/annotated_paths_GPT4o/{split}')
with open(f'{processed_dir}/{split}/metadata_{split}.pkl', 'rb') as f:
    metadata_list = pickle.load(f)
print(len(text_labels), len(metadata_list), len(retrieval_list))
assert len(text_labels) == len(metadata_list) == len(retrieval_list)
labeled_topic_relation_path = f'data_files/{name}/topic_relation_candidates/{split}_results.pkl'
post_dataset = ProcessedDiskDataset(post_processed_dir, split, raw_dataset, labeled_topic_relation_path,
                                    retrieval_list=retrieval_list, metadata_list=metadata_list, text_labels=text_labels)
print(post_dataset[0])
print('post-processed test samples:', len(post_dataset), '->', post_processed_dir)
