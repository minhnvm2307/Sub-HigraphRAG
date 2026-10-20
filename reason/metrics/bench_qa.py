"""QA tables of BENCHMARK.md (10-13): every reasoning run on disk, scored the same way (metrics/paper_score.py: Hit of the
original eval on raw responses, Macro-F1 of the corrected eval with the original substring match).

Question sets:
  full    the questions of the official SubgraphRAG prediction files (= what both papers report): WebQSP 1,628 (the 11
          other test ids have no gold answer), CWQ 3,531
  subset  full questions with at least one gold answer entity inside the KG subgraph (a_entity_in_graph): WebQSP 1,559,
          CWQ 2,848; retrieval metrics are only defined on these
A run that holds fewer questions is scored on the ones it has (n reported), and the official SubgraphRAG @100 file
(bf16, authors) is scored on exactly the same ids next to it.
  python -m metrics.bench_qa   (from reason/) -> results/bench_qa.json
"""
import json, os, tempfile
import torch
from metrics.paper_score import score

B = 'results/KGQA'
OFF = {'webqsp': f'{B}/webqsp/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl',
       'cwq': f'{B}/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl'}
W, C = f'{B}/webqsp/SubgraphRAG', f'{B}/cwq/SubgraphRAG'
CR = 'custom_custom_retrieval'
# (key, dataset, retriever, train, budget, reasoner, prediction format, file)
RUNS = [
    # WebQSP test
    ('w_sgr100_off', 'webqsp', 'SubgraphRAG official retrieval', 'WebQSP', 'top-100', 'Llama-3.1-8B bf16 (authors\' file)', 'raw', f'{W}/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl'),
    ('w_sgr500_off', 'webqsp', 'SubgraphRAG official retrieval', 'WebQSP', 'top-500', 'Llama-3.1-8B bf16 (authors\' file)', 'raw', f'{W}/Meta-Llama-3.1-8B-Instruct/scored_500-sys_icl_dc-0-thres_0.0-test-predictions.jsonl'),
    ('w_sgr100_t4', 'webqsp', 'SubgraphRAG official retrieval', 'WebQSP', 'top-100', 'Llama-3.1-8B fp16 vLLM T4 (EXP-25, calibration)', 'raw', f'{W}/vllm:llama31-8b/scored_100-sys_icl_dc-0-thres_0.0-test-custom_wq_sgr_official-raw-predictions.jsonl'),
    ('w_relrag_t4', 'webqsp', 'REL-RAG repro (P, path_bf)', 'WebQSP', 'top-50', 'Llama-3.1-8B fp16 vLLM T4 (EXP-25)', 'raw', f'{W}/vllm:llama31-8b/scored_50-sys_icl_dc-0-thres_0.0-test-custom_wq_relrag_p-raw-predictions.jsonl'),
    ('w_aer_t4', 'webqsp', 'AER (DDE, AER-trained)', 'WebQSP', 'top-50', 'Llama-3.1-8B fp16 vLLM T4 (EXP-25)', 'raw', f'{W}/vllm:llama31-8b/scored_50-sys_icl_dc-0-thres_0.0-test-custom_wq_aer-raw-predictions.jsonl'),
    ('w_rbf50', 'webqsp', 'RBF+nb+mm', 'WebQSP', 'top-50', 'Llama-3.1-8B fp8 (DeepInfra via OpenRouter)', 'raw*', f'{W}/openrouter/scored_50-sys_icl_dc-0-thres_0.0-test-{CR}_webqsp_rbfnb_mm_top100-raw-predictions.jsonl'),
    ('w_rbf100', 'webqsp', 'RBF+nb+mm', 'WebQSP', 'top-100', 'Llama-3.1-8B fp8 (DeepInfra via OpenRouter)', 'raw*', f'{W}/openrouter/scored_100-sys_icl_dc-0-thres_0.0-test-{CR}_webqsp_rbfnb_mm_top100-raw-predictions.jsonl'),
    ('w_q4_sgr100', 'webqsp', 'SubgraphRAG official retrieval', 'WebQSP', 'top-100', 'Llama-3.1-8B Q4 ollama (EXP-17)', 'ans', f'{W}/ollama:llama/scored_100-sys_icl_dc-0-thres_0.0-test-custom_scored_triples_webqsp_240912_unidir_test-predictions.jsonl'),
    ('w_q4_sgr50', 'webqsp', 'SubgraphRAG official retrieval', 'WebQSP', 'top-50', 'Llama-3.1-8B Q4 ollama (EXP-17)', 'ans', f'{W}/ollama:llama/scored_50-sys_icl_dc-0-thres_0.0-test-custom_scored_triples_webqsp_240912_unidir_test-predictions.jsonl'),
    ('w_q4_relrag_old', 'webqsp', 'REL-RAG repro cũ (P, nhãn gemma, path_bf)', 'WebQSP', 'top-50', 'Llama-3.1-8B Q4 ollama (EXP-17)', 'ans', f'{W}/ollama:llama/scored_50-sys_icl_dc-0-thres_0.0-test-custom_path_llm_noselfloop_path_bf_retrieval_result_50-predictions.jsonl'),
    ('w_q4_relrag_free', 'webqsp', 'REL-RAG repro (P, path_free, ~43 triple)', 'WebQSP', 'path_free (~43)', 'Llama-3.1-8B Q4 ollama (EXP-17)', 'ans', f'{W}/ollama:llama/scored_1000-sys_icl_dc-0-thres_0.0-test-custom_retrieval_test_path_free_retrieval_result_free-predictions.jsonl'),
    # CWQ test
    ('c_sgr100_off', 'cwq', 'SubgraphRAG official retrieval', 'CWQ', 'top-100', 'Llama-3.1-8B bf16 (authors\' file)', 'raw', f'{C}/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl'),
    ('c_sgr500_off', 'cwq', 'SubgraphRAG official retrieval', 'CWQ', 'top-500', 'Llama-3.1-8B bf16 (authors\' file)', 'raw', f'{C}/Meta-Llama-3.1-8B-Instruct/scored_500-sys_icl_dc-0-thres_0.0-test-predictions.jsonl'),
    ('c_a1_relrag', 'cwq', 'REL-RAG repro (P, path_bf)', 'WebQSP', 'top-50', 'Llama-3.1-8B bf16 vLLM 4090 (EXP-26, A1)', 'raw', f'{C}/vllm-bf16-4090/scored_50-sys_icl_dc-0-thres_0.0-test-custom_a1_a1_relrag_p-raw-predictions.jsonl'),
    ('c_a1_dde', 'cwq', 'DDE (SubgraphRAG repro)', 'WebQSP', 'top-50', 'Llama-3.1-8B bf16 vLLM 4090 (EXP-26, A1)', 'raw', f'{C}/vllm-bf16-4090/scored_50-sys_icl_dc-0-thres_0.0-test-custom_a1_a1_dde-raw-predictions.jsonl'),
    ('c_a1_ddere', 'cwq', 'DDE + neo lại lúc test (không train)', 'WebQSP', 'top-50', 'Llama-3.1-8B bf16 vLLM 4090 (EXP-26, A1)', 'raw', f'{C}/vllm-bf16-4090/scored_50-sys_icl_dc-0-thres_0.0-test-custom_a1_a1_dde_reanchor-raw-predictions.jsonl'),
    ('c_a1_aer', 'cwq', 'AER (DDE, AER-trained)', 'WebQSP', 'top-50', 'Llama-3.1-8B bf16 vLLM 4090 (EXP-26, A1)', 'raw', f'{C}/vllm-bf16-4090/scored_50-sys_icl_dc-0-thres_0.0-test-custom_a1_a1_aer-raw-predictions.jsonl'),
    ('c_a2_wl', 'cwq', 'WL + ADR + AER', 'WebQSP', 'top-50', 'Llama-3.1-8B bf16 vLLM 4090 (EXP-30, A2)', 'raw', f'{C}/vllm-bf16-4090-full/scored_50-sys_icl_dc-0-thres_0.0-test-custom_a2_a2_wl_adr_aer-raw-predictions.jsonl'),
    ('c_rbfw50', 'cwq', 'RBF+nb+mm', 'WebQSP', 'top-50', 'Llama-3.1-8B fp8 (DeepInfra)', 'raw', f'{C}/deepinfra/scored_50-sys_icl_dc-0-thres_0.0-test-{CR}_cwq_rbfnbwebqsp_mm_top100-raw-predictions.jsonl'),
    ('c_rbfc50', 'cwq', 'RBF+nb+mm', 'CWQ', 'top-50', 'Llama-3.1-8B fp8 (DeepInfra via OpenRouter)', 'raw*', f'{C}/openrouter/scored_50-sys_icl_dc-0-thres_0.0-test-{CR}_cwq_rbfnbcwq_mm_top100-raw-predictions.jsonl'),
    ('c_rbfc100', 'cwq', 'RBF+nb+mm', 'CWQ', 'top-100', 'Llama-3.1-8B fp8 (DeepInfra)', 'raw', f'{C}/deepinfra/scored_100-sys_icl_dc-0-thres_0.0-test-{CR}_cwq_rbfnbcwq_mm_top100-raw-predictions.jsonl'),
    ('c_q4_pmh', 'cwq', 'RT Path+Memory hybrid (bản SubgraphRAG cũ, 09-06)', 'CWQ', 'top-100', 'Llama-3.1-8B Q4 ollama', 'ans', f'{C}/ollama:llama/scored_100-sys_icl_dc-0-thres_0.0-test-custom_cwq_rt_path_memory_hybrid_bce_Sep06-03:05:00-predictions.jsonl'),
]


def load(p):
    return [json.loads(l) for l in open(p)]


def subset_ids(ds):
    ref = torch.load(f'scored_triples/{"webqsp_240912_unidir_test" if ds == "webqsp" else "cwq_240907_unidir_test"}.pth', weights_only=False)
    return {i for i, v in ref.items() if v['a_entity_in_graph']}


def scored(rows, ds, name, tmp):
    """paper_score on `rows` (written under .../<ds>/ so the evaluator picks the right reference)"""
    d = os.path.join(tmp, ds); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return score(p)


def main():
    full = {ds: [r['id'] for r in load(OFF[ds])] for ds in OFF}
    sub = {ds: subset_ids(ds) for ds in OFF}
    off_rows = {ds: {r['id']: r for r in load(OFF[ds])} for ds in OFF}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, ds, ret, train, budget, reasoner, fmt, path in RUNS:
            rows = {r['id']: r for r in load(path)}
            ids = [i for i in full[ds] if i in rows]
            name = os.path.basename(path).replace('custom_', 'c_')
            s = scored([rows[i] for i in ids], ds, name, tmp)
            ref = scored([off_rows[ds][i] for i in ids], ds, 'scored_100-sys_icl_dc-0-thres_0.0-test-ref-predictions.jsonl', tmp)
            out[key] = dict(dataset=ds, retriever=ret, train=train, setting='ID' if train.lower() == ds else 'OOD',
                            budget=budget, reasoner=reasoner, pred_format=fmt, file=path,
                            n_full=len(ids), n_full_all=len(full[ds]),
                            full=s['full'][:3], subset=s['subset'][:3], n_subset=s['subset'][3],
                            ref_same_ids_full=ref['full'][:3], ref_same_ids_subset=ref['subset'][:3])
            print(key, out[key]['n_full'], [round(x, 2) for x in out[key]['full']], [round(x, 2) for x in out[key]['subset']], flush=True)
    json.dump(out, open('results/bench_qa.json', 'w'), indent=1)
    print('-> results/bench_qa.json')


if __name__ == '__main__':
    main()
