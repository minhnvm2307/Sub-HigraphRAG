"""Step 3 — test-time labels the authors' pipeline requires, in the authors' file formats.

RAPL's decoding starts from data.topic_labels (src/model/Trainerv3.py decoding, way='normal'), which
PostProcessedDataset._update_topic_labels builds from (A) LLM-labelled topic relations
(data_files/<ds>/topic_relation_candidates/<split>_results.pkl, RelationPathTargeting.py) and (C) the first nodes of the
labelled rational paths (data/<ds>/raw/annotated_paths_GPT4o/<split>/sample_<i>.txt, FindRationalPath.py). The released
scripts produce both for the test split with GPT-4o; they need the gold answers, so this is test-label use by design.

Label source chosen (user decision 10-10): our hybrid GPT-4o + DeepSeek rational paths exist only for WebQSP train/val
(relrag/data/labels_hybrid_gpt4o_deepseek/{train,val}.jsonl), not for CWQ test -> not applicable -> SP labels:
  rational paths  = the shortest relation sequences among the authors' own candidate paths of the sample
                    (RetrieverDataset reasoning_paths: every q -> answer path within shortest + 2 hops), up to 10
                    (PostProcessedDataset keeps max_paths = 10)
  topic relations = the first relation of each of those paths (the role RelationPathTargeting plays)
One file per sample, for every sample: KGDataset._collect_annotated_paths lists only existing files, so a missing file
would shift every later sample's labels.
Run from RAPL/:   cd RAPL && python ../rapl_faithful/s3_make_sp_labels.py
"""
import os, pickle
assert os.path.basename(os.getcwd()) == 'RAPL', 'run from the RAPL/ directory'
DS, SPLIT, MAXP = 'cwq', 'test', 10
recs = pickle.load(open(f'data_files/{DS}/processed/{SPLIT}_retrieval.pkl', 'rb'))
lab_dir = f'data/{DS}/raw/annotated_paths_GPT4o/{SPLIT}'
os.makedirs(lab_dir, exist_ok=True)
os.makedirs(f'data_files/{DS}/topic_relation_candidates', exist_ok=True)
topic_rel, n_lab, n_none = [], 0, 0
for i, s in enumerate(recs):
    paths = [tuple(p) for p in (s.get('reasoning_paths') or []) if len(p) > 0]
    if paths:
        m = min(len(p) for p in paths)
        sp = list(dict.fromkeys(p for p in paths if len(p) == m))[:MAXP]
    else:
        sp = []
    n_lab += bool(sp); n_none += not sp
    body = '\n'.join(f'{j + 1}. [{", ".join(p)}]' for j, p in enumerate(sp))
    with open(f'{lab_dir}/sample_{i}.txt', 'w', encoding='utf-8') as f:
        f.write(f'Question: {s["question"]}\nLabel source: shortest candidate paths (SP), not GPT-4o.\n'
                f'The rational paths are:\n{body}\n')
    firsts = list(dict.fromkeys(p[0] for p in sp))
    topic_rel.append({'response_text': f'<Solution:> [{", ".join(firsts)}]'})
pickle.dump(topic_rel, open(f'data_files/{DS}/topic_relation_candidates/{SPLIT}_results.pkl', 'wb'))
print(f'{len(recs)} samples: {n_lab} with SP labels, {n_none} without a q -> answer path (empty label, topic_labels from '
      f'sampled candidates only); files in {lab_dir}')
