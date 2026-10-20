"""External multi-hop KGQA benchmarks (GTSQA, MetaQA-3hop) in the format of the SubgraphRAG/QR-DDE retrievers, for
zero-shot evaluation of WebQSP-trained retrievers (docs/09, claim C13).

  convert  parquet (per-question subgraph, RoG-like) -> data_ext/<ds>/test.pkl: records with h/r/t ids, q_ent, a_ent,
           names, relations, and (GTSQA) the gold answer-subgraph triples, n_hops, seeds, query-graph type, test type
  embed    GTE-large-en-v1.5 (same encoder/pooling as retrieve/src/model/text_encoders/gte_large_en.py) over the
           *unique* entity names / relations of the whole split -> one global table + per-question index arrays
           (GTSQA has 1.8M unique entities; per-question copies as for CWQ would take ~60GB)
GlobalEmb exposes .index and .get(sid) -> (q, ent, rel) like qrdde.SubsetEmb.

  python ext_data.py convert gtsqa ; conda run -n gte-emb python ext_data.py embed gtsqa
"""
import os, re, sys, pickle, glob
import numpy as np

ROOT = '/home/minhnv/Lab/SubgraphRAG/data_ext'
SRC = {'gtsqa': [f'{ROOT}/gtsqa_test.parquet'], 'metaqa3': sorted(glob.glob(f'{ROOT}/metaqa3_test*.parquet')),
       'grailqa': f'{ROOT}/grailqa_raw'}
MID = re.compile(r'^[mg]\.[0-9a-z_]+$')


def convert_grailqa_oda(max_triples=30000):
    """GrailQA set of the REL-RAG paper (§5.1, App. D): the 1,000 questions of the ODA repo (lanjiuqing64/KGdata
    QAdata/graliqa.json) with the same ODA 2-hop Freebase subgraphs; the paper keeps 874. Same construction as
    convert_grailqa; adds the first domain (paper Fig. 4 reports the top-5 domains)."""
    import json
    rows = json.load(open(f'{ROOT}/grailqa_oda/graliqa.json'))
    d = SRC['grailqa']
    trip = json.load(open(f'{d}/hop_2_Freebase/grailqa_triples_2_hop_all.json'))
    lab = json.load(open(f'{d}/hop_2_Freebase/grailqa_labels_2_hop_all.json'))
    recs, stat = [], {'no_topic': 0, 'no_answer': 0, 'too_big': 0}
    for row in rows:
        tops = [n['id'] for n in row['graph_query']['nodes'] if n['node_type'] == 'entity' and n['id'] in trip]
        rec = _grail_record(row['qid'], row['question'], tops, [a['answer_argument'] for a in row['answer']], trip, lab,
                            max_triples, stat)
        if rec is None:
            continue
        rec.update({'n_hops': int(row['num_edge']), 'level': row['level'], 'function': row['function'],
                    'domain': (row.get('domains') or ['-'])[0]})
        recs.append(rec)
    os.makedirs(f'{ROOT}/grailqa_oda', exist_ok=True)
    pickle.dump(recs, open(f'{ROOT}/grailqa_oda/test.pkl', 'wb'))
    print('grailqa (ODA set)', len(rows), 'questions ->', len(recs), 'kept;', stat)


def convert_grailqa_paper(max_triples=30000):
    """GrailQA set of the REL-RAG paper as App. D describes it (EXP-37): questions of the GrailQA *train* split
    (HF dki-lab/grail_qa, refs/convert/parquet, 44,337 questions) whose topic entities have an ODA 2-hop Freebase
    subgraph (CCaccept/Subgraph hop_2_Freebase, 879 topic MIDs); a question is kept if at least one topic has one
    (995 questions; the paper reports 874 after its own filtering). Same record construction as convert_grailqa_oda:
    subgraph = union of the topics' subgraphs, dropped if no answer is in it or it exceeds max_triples."""
    import json, pandas as pd
    d = SRC['grailqa']
    trip = json.load(open(f'{d}/hop_2_Freebase/grailqa_triples_2_hop_all.json'))
    lab = json.load(open(f'{d}/hop_2_Freebase/grailqa_labels_2_hop_all.json'))
    df = pd.read_parquet(f'{d}/train/0000.parquet')
    recs, stat, n_cand = [], {'no_topic': 0, 'no_answer': 0, 'too_big': 0}, 0
    for _, row in df.iterrows():
        nodes = row['graph_query']['nodes']
        tops_all = [i for i, t in zip(nodes['id'], nodes['node_type']) if t == 'entity']
        tops = [t for t in tops_all if t in trip]
        if not tops:
            continue
        n_cand += 1
        rec = _grail_record(row['qid'], row['question'], tops, list(row['answer']['answer_argument']), trip, lab,
                            max_triples, stat)
        if rec is None:
            continue
        rec.update({'n_hops': int(row['num_edge']), 'function': row['function'],
                    'domain': (list(row['domains']) or ['-'])[0], 'all_topics_covered': len(tops) == len(tops_all)})
        recs.append(rec)
    os.makedirs(f'{ROOT}/grailqa_paper', exist_ok=True)
    pickle.dump(recs, open(f'{ROOT}/grailqa_paper/test.pkl', 'wb'))
    print('grailqa (paper set)', len(df), 'train questions ->', n_cand, 'with an ODA topic ->', len(recs), 'kept;', stat)


def _grail_record(qid, question, tops, ans, trip, lab, max_triples, stat):
    if not tops:
        stat['no_topic'] += 1; return None
    T, names = [], {}
    for tp in tops:
        T += [tuple(x) for x in trip[tp]]
        names.update(lab.get(tp, {}))
    T = list(dict.fromkeys(T))
    if len(T) > max_triples:
        stat['too_big'] += 1; return None
    ents = list(dict.fromkeys([x for h, _, t in T for x in (h, t)]))
    text_e = [e for e in ents if e in names or not MID.match(e)]
    non_text = [e for e in ents if not (e in names or not MID.match(e))]
    order = text_e + non_text; eid = {e: k for k, e in enumerate(order)}
    rels = list(dict.fromkeys(r for _, r, _ in T)); rid = {r: k for k, r in enumerate(rels)}
    a_ent = sorted({eid[a] for a in ans if a in eid})
    q_ent = sorted({eid[t] for t in tops if t in eid})
    if not a_ent or not q_ent:
        stat['no_answer'] += 1; return None
    disp = [names.get(e, e) for e in order]
    return {'id': str(qid), 'question': question,
            'h': np.array([eid[h] for h, _, _ in T], dtype=np.int64), 'r': np.array([rid[r] for _, r, _ in T], dtype=np.int64),
            't': np.array([eid[t] for _, _, t in T], dtype=np.int64), 'q_ent': q_ent, 'a_ent': a_ent,
            'n_ent': len(order), 'n_text': len(text_e), 'names': disp, 'rels': rels,
            'a_names': [disp[k] for k in a_ent], 'n_seeds': len(q_ent)}


def convert_grailqa(max_triples=30000):
    """GrailQA dev (Hieuman/grail_qa validation, with answers) + Freebase 2-hop subgraphs per topic entity
    (CCaccept/Subgraph hop_2_Freebase, from the ODA paper). Subgraph of a question = union of its topic entities'
    2-hop subgraphs. Entities with a label are text entities (ordered first); unlabeled mids are CVT / non-text nodes,
    which the DDE-family scorers give the shared non-text embedding, as for WebQSP/CWQ (n_text marks the split)."""
    import json, pandas as pd
    d = SRC['grailqa']
    df = pd.read_parquet(f'{d}/data/validation-00000-of-00001.parquet')
    trip = json.load(open(f'{d}/hop_2_Freebase/grailqa_triples_2_hop_all.json'))
    lab = json.load(open(f'{d}/hop_2_Freebase/grailqa_labels_2_hop_all.json'))
    recs, stat = [], {'no_topic': 0, 'no_answer': 0, 'too_big': 0}
    for _, row in df.iterrows():
        g = row['graph_query']['nodes']
        tops = [i for t, i in zip(g['node_type'], g['id']) if t == 'entity' and i in trip]
        if not tops:
            stat['no_topic'] += 1; continue
        T = []
        names = {}
        for tp in tops:
            T += [tuple(x) for x in trip[tp]]
            names.update(lab.get(tp, {}))
        T = list(dict.fromkeys(T))
        if len(T) > max_triples:
            stat['too_big'] += 1; continue
        ents = list(dict.fromkeys([x for h, _, t in T for x in (h, t)]))
        text_e = [e for e in ents if e in names or not MID.match(e)]       # labeled mids and literal values
        non_text = [e for e in ents if not (e in names or not MID.match(e))]
        order = text_e + non_text; eid = {e: k for k, e in enumerate(order)}
        rels = list(dict.fromkeys(r for _, r, _ in T)); rid = {r: k for k, r in enumerate(rels)}
        ans = list(row['answer']['answer_argument'])
        a_ent = sorted({eid[a] for a in ans if a in eid})
        q_ent = sorted({eid[t] for t in tops if t in eid})
        if not a_ent or not q_ent:
            stat['no_answer'] += 1; continue
        disp = [names.get(e, e) for e in order]
        recs.append({'id': str(row['qid']), 'question': row['question'],
                     'h': np.array([eid[h] for h, _, _ in T], dtype=np.int64), 'r': np.array([rid[r] for _, r, _ in T], dtype=np.int64),
                     't': np.array([eid[t] for _, _, t in T], dtype=np.int64), 'q_ent': q_ent, 'a_ent': a_ent,
                     'n_ent': len(order), 'n_text': len(text_e), 'names': disp, 'rels': rels,
                     'a_names': [disp[k] for k in a_ent], 'n_hops': int(row['num_edge']), 'n_seeds': len(q_ent),
                     'level': row['level'], 'function': row['function']})
    os.makedirs(f'{ROOT}/grailqa', exist_ok=True)
    pickle.dump(recs, open(f'{ROOT}/grailqa/test.pkl', 'wb'))
    print('grailqa dev', len(df), 'questions ->', len(recs), 'kept;', stat)
WD = re.compile(r'\s*\((?:Q|P)\d+\)\s*$')


def text(x):
    """name used for the text embedding: drop the Wikidata id suffix ' (Q123)' / ' (P58)'"""
    return WD.sub('', x)


def convert(ds):
    if ds == 'grailqa':
        return convert_grailqa()
    if ds == 'grailqa_oda':
        return convert_grailqa_oda()
    if ds == 'grailqa_paper':
        return convert_grailqa_paper()
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in SRC[ds]], ignore_index=True)
    recs = []
    for _, row in df.iterrows():
        G = [tuple(t) for t in row['graph']]
        ents, rels = {}, {}
        H, R, T = [], [], []
        for h, r, t in G:
            H.append(ents.setdefault(h, len(ents))); R.append(rels.setdefault(r, len(rels))); T.append(ents.setdefault(t, len(ents)))
        names = list(ents); rel_list = list(rels)
        if ds == 'gtsqa':
            seeds = list(row['seed_entities'])
            qid2name = {n.rsplit('(', 1)[-1].rstrip(')'): n for n in names}
            answers = [qid2name[q] for q in row['all_answers_wikidata'] if q in qid2name]
            extra = {'n_hops': int(row['n_hops']), 'n_seeds': len(seeds), 'iso': row['minimal_graph_isomorphism'],
                     'test_type': list(row['test_type']), 'gold': [tuple(t) for t in row['answer_subgraph']]}
        else:
            seeds = list(row['q_entity']); answers = [a for a in row['a_entity'] if a in ents]; extra = {'n_hops': 3, 'n_seeds': len(seeds)}
        q_ent = [ents[s] for s in seeds if s in ents]; a_ent = [ents[a] for a in answers]
        if not q_ent or not a_ent:
            continue
        recs.append({'id': str(row['id']), 'question': row['question'], 'h': np.array(H, dtype=np.int64), 'r': np.array(R, dtype=np.int64),
                     't': np.array(T, dtype=np.int64), 'q_ent': sorted(set(q_ent)), 'a_ent': sorted(set(a_ent)), 'n_ent': len(names),
                     'names': names, 'rels': rel_list, 'a_names': answers, **extra})
    os.makedirs(f'{ROOT}/{ds}', exist_ok=True)
    pickle.dump(recs, open(f'{ROOT}/{ds}/test.pkl', 'wb'))
    print(ds, len(df), 'questions ->', len(recs), 'kept (seed and answer in graph)')


def embed(ds, bs=32):
    import torch, torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    recs = pickle.load(open(f'{ROOT}/{ds}/test.pkl', 'rb'))
    ent_vocab, rel_vocab = {}, {}
    for r in recs:
        for n in r['names'][:r.get('n_text', len(r['names']))]: ent_vocab.setdefault(text(n), len(ent_vocab))
        for n in r['rels']: rel_vocab.setdefault(text(n), len(rel_vocab))
    tok = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, unpad_inputs=True,
                                      use_memory_efficient_attention=True).to('cuda').eval()
    out = f'{ROOT}/{ds}/emb'; os.makedirs(out, exist_ok=True)

    @torch.no_grad()
    def enc(texts, path):
        if os.path.exists(path):
            return
        mm = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.float16, shape=(len(texts), 1024))
        order = np.argsort([len(t) for t in texts])          # length-sorted batches: much less padding
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            b = tok([texts[j] for j in idx], max_length=8192, padding=True, truncation=True, return_tensors='pt').to('cuda')
            e = F.normalize(model(**b).last_hidden_state[:, 0], p=2, dim=1)
            mm[idx] = e.half().cpu().numpy()
            if i // bs % 200 == 0:
                print(f'  {os.path.basename(path)} {i}/{len(texts)}', flush=True)
        mm.flush(); del mm; os.replace(path + '.tmp', path)
    enc(list(ent_vocab), f'{out}/ent_table.npy')
    enc(list(rel_vocab), f'{out}/rel_table.npy')
    enc([r['question'] for r in recs], f'{out}/q.npy')
    index = {r['id']: (k, np.array([ent_vocab[text(n)] for n in r['names'][:r.get('n_text', len(r['names']))]], dtype=np.int32),
                       np.array([rel_vocab[text(n)] for n in r['rels']], dtype=np.int32)) for k, r in enumerate(recs)}
    pickle.dump(index, open(f'{out}/index.pkl', 'wb'))
    print(ds, 'entities', len(ent_vocab), 'relations', len(rel_vocab), 'questions', len(recs))


class GlobalEmb:
    """same .index / .get(sid) -> (q, ent, rel) contract as qrdde.SubsetEmb, backed by global memmapped tables"""
    def __init__(self, ds, d=None):
        import torch
        self.torch = torch
        d = d or f'{ROOT}/{ds}/emb'
        self.index = pickle.load(open(f'{d}/index.pkl', 'rb'))
        self.q, self.ent, self.rel = (np.load(f'{d}/{n}.npy', mmap_mode='r') for n in ('q', 'ent_table', 'rel_table'))

    @classmethod
    def from_dir(cls, d):
        return cls(None, d)

    def get(self, sid):
        k, ei, ri = self.index[sid]
        t = self.torch.from_numpy
        return t(np.array(self.q[k:k + 1])), t(np.asarray(self.ent[ei])), t(np.asarray(self.rel[ri]))


if __name__ == '__main__':
    {'convert': convert, 'embed': embed}[sys.argv[1]](sys.argv[2])
