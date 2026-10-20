"""Step 1b — data_files/cwq/processed/test.pkl in the authors' format, built by the authors' own EmbInferDataset
(src/dataset/emb.py, exactly as RAPL/emb.py calls it for the test split: skip_no_topic=False, skip_no_ans=False) from HF
rmanluo/RoG-cwq. The authors' records add 'id2entities' / 'id2relations' to the SubgraphRAG format, which
RetrieverDataset needs. Only the text-embedding part of emb.py is not re-run: the GTE embeddings of the SubgraphRAG CWQ
test export are reused, which is valid only if ids and entity / relation order are identical -- checked below.
Run from RAPL/:   cd RAPL && python ../rapl_faithful/s1b_build_processed.py
"""
import os, pickle, sys
sys.path.insert(0, os.getcwd())
assert os.path.basename(os.getcwd()) == 'RAPL', 'run from the RAPL/ directory'
from datasets import load_dataset
from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

ROOT = os.environ.get('SGRAG_ROOT', '/home/minhnv/Lab/SubgraphRAG')
out = 'data_files/cwq/processed/test.pkl'
if os.path.islink(out):
    os.remove(out)                       # an earlier setup linked the SubgraphRAG file here (no id2entities)
config = load_yaml('configs/emb/gte-large-en-v1.5/cwq.yaml')
ids = set(l.strip() for l in open(config['entity_identifier_file']))
test_set = load_dataset('rmanluo/RoG-cwq', split='test')
ds = EmbInferDataset(test_set, ids, out, skip_no_topic=False, skip_no_ans=False)
recs = pickle.load(open(out, 'rb'))
sg = pickle.load(open(f'{ROOT}/retrieve/data_files/cwq/processed/test.pkl', 'rb'))
bad = sum(a['id'] != b['id'] or a['text_entity_list'] != b['text_entity_list'] or a['relation_list'] != b['relation_list']
          or a['non_text_entity_list'] != b['non_text_entity_list'] for a, b in zip(recs, sg))
print(f'authors test.pkl: {len(recs)} records; vs SubgraphRAG export ({len(sg)}): {bad} differ in id / entity / relation order')
assert len(recs) == len(sg) and bad == 0, 'GTE embeddings of the SubgraphRAG export cannot be reused'
assert 'id2entities' in recs[0]
