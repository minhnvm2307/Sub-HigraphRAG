"""SubgraphRAG vs RAPL (authors' released checkpoints) vs REL-RAG (our reproduction), scored by SubgraphRAG's own
retrieval evaluator, unmodified: retrieve/eval.py (ans / shortest-path / GPT triple recall and precision at
k = 50,100,200,400,500).

eval.py reads the labels from the export itself (a_entity_in_graph, target_relevant_triples), which only SubgraphRAG
exports carry. For every other method the export is rebuilt with its own scored_triples and the labels of the
official SubgraphRAG export of the same test set, question by question; a question missing from a method gets an
empty list (eval.py skips it, as it skips SubgraphRAG's own empty subgraphs). Within a setting every method is
scored on the same questions: those present in every method's export (our SubgraphRAG repro exports hold only the
questions with an answer in the graph). Exports hold at most 500 triples
(REL-RAG / RAPL: path best-first decoding at budget 500, k < 500 = prefix, verified prefix-consistent in EXP-31b).
  python sg_eval_compare.py > runs/sg_eval_compare.md
"""
import os, subprocess, sys, tempfile
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
R = f'{ROOT}//relrag/runs'
Q = f'{ROOT}/retrieve/exp_qrdde/runs'
REF = {'webqsp': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
       'cwq': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth'}
SETTINGS = [
    ('WebQSP (train WebQSP, test WebQSP)', 'webqsp', [
        ('SubgraphRAG (official ckpt)', REF['webqsp']),
        ('RAPL (authors, webqsp.pt)', f'{R}/authors_ckpt/webqsp_model/retrieval_test_path_bf/retrieval_result_500.pth'),
        ('REL-RAG (our repro)', f'{R}/path_codefaithful/retrieval_test_path_bf/retrieval_result_500.pth')]),
    ('CWQ (train CWQ, test CWQ)', 'cwq', [
        ('SubgraphRAG (official ckpt)', REF['cwq']),
        ('RAPL (authors, cwq.pt)', f'{R}/authors_ckpt/cwq_model/retrieval_cwq_test_path_bf/retrieval_result_500.pth')]),
    ('WebQSP -> CWQ (train WebQSP, test CWQ)', 'cwq', [
        ('SubgraphRAG (our repro, train WebQSP)', f'{Q}/webqsp_dde/adr_cwq_base.pth'),
        ('RAPL (authors, webqsp.pt)', f'{R}/authors_ckpt/webqsp_model/retrieval_cwq_test_path_bf/retrieval_result_500.pth'),
        ('REL-RAG (our repro)', f'{R}/path_codefaithful/retrieval_cwq_test_path_bf/retrieval_result_500.pth')]),
    ('CWQ -> WebQSP (train CWQ, test WebQSP)', 'webqsp', [
        ('SubgraphRAG (our repro, train CWQ)', f'{Q}/cwq_dde/adr_webqsp_base.pth'),
        ('RAPL (authors, cwq.pt)', f'{R}/authors_ckpt/cwq_model/retrieval_test_path_bf/retrieval_result_500.pth')]),
]

def rebuild(path, ref):
    d = torch.load(path, weights_only=False)
    out = {}
    for i, r in ref.items():
        st = [tuple(x[:3]) + (float(x[3]) if len(x) > 3 else 0.0,) for x in d[i]['scored_triples']] if i in d else []
        out[i] = {'scored_triples': st, 'a_entity_in_graph': r['a_entity_in_graph'],
                  'target_relevant_triples': r['target_relevant_triples']}
    return out, sum(i in d for i in ref)

for title, ds, methods in SETTINGS:
    ref = torch.load(REF[ds], weights_only=False)
    common = set(ref)
    for _, path in methods:
        if os.path.exists(path):
            common &= set(torch.load(path, weights_only=False))
    ref = {i: ref[i] for i in ref if i in common}
    print(f'## {title}\n\nScored on the {len(ref)} questions present in every export of this setting.\n')
    for name, path in methods:
        if not os.path.exists(path):
            print(f'### {name}\n\n(missing export: {path})\n'); continue
        pred, cov = rebuild(path, ref)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(pred, f.name)
        res = subprocess.run([sys.executable, 'eval.py', '-d', ds, '-p', f.name],
                             cwd=f'{ROOT}/retrieve', capture_output=True, text=True)
        os.unlink(f.name)
        print(f'### {name}\n\n{cov}/{len(ref)} test questions in the export\n\n```\n{res.stdout.strip() or res.stderr[-800:]}\n```\n')
