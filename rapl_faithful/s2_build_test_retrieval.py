"""Step 2 — build data_files/cwq/processed/test_retrieval.pkl with the authors' own RetrieverDataset, exactly the test
block of RAPL/MakeTrainData.py (RetrieverDataset(config, split='test', skip_no_path=False)). MakeTrainData.py itself is
not run because it first processes the whole CWQ train and val sets, which this evaluation does not need.
Run from RAPL/:   cd RAPL && python ../rapl_faithful/s2_build_test_retrieval.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
assert os.path.basename(os.getcwd()) == 'RAPL', 'run from the RAPL/ directory'
from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset
from src.setup import set_seed

config = load_yaml('configs/retriever/cwq.yaml')            # the authors' config (seed 42)
set_seed(config['env']['seed'])
ds = RetrieverDataset(config=config, split='test', skip_no_path=False)
print('test samples:', len(ds.processed_dict_list), '->', 'data_files/cwq/processed/test_retrieval.pkl')
