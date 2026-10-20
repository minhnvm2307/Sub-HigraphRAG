"""LLM reasoning with REL-RAG's path-format input, as the authors' code does it:
  input   RAPL/model_inference.py::write_question_paths_to_txt — a header with the question, then the retrieved
          paths, one per line as "e1 -> r1 -> e2 -> r2 -> e3", in decreasing path probability
          (infer.py exports them as `paths`)
  prompt  RAPL/generate_answers_llama31_8b.py — that text + "\\n" + post_str ("think step by step ... 1. The answer
          is: ..."), one user turn, Llama-3.1-8B-Instruct chat template
  scoring the paper's protocol (SubgraphRAG / RoG eval): Hit from reason/metrics/evaluate_results.py, Macro-F1 from
          evaluate_results_corrected.py, on every test question
Deviations, forced by the hardware (GTX 1080 Ti, 11 GB, no bf16): llama3.1 8B Q4 on ollama instead of bf16
transformers, and greedy decoding (ollama has no beam search) instead of num_beams=5 / length_penalty=1.2;
max_new_tokens=500 and repetition_penalty=1.25 are kept. The authors release no answer parser; the text after
"The answer is:" is split on commas/semicolons/"and"/newlines into one "ans: <x>" line per answer, the format
the eval scripts expect (one line per predicted answer, so precision is not inflated).

  python reason_paths.py --retrieval runs/<run>/retrieval_test_path_free/retrieval_result_free.pth [--limit 5]
"""
import argparse, json, os, re, sys
import requests
import torch
from tqdm import tqdm

ROOT = '/home/minhnv/Lab/SubgraphRAG'
POST_STR = """
    Answer the question, also tell me if you used the provided knowledge above.
    Now think step by step and answer this question. Use the following template for your final answers:
    <Some thinking process here>
    <Solution>
    1. The answer is: <Your answer here>
    2. I used the provided knowledge: <Yes/No>, followed by your explanation
    """


def user_text(question, paths):
    header = (f"Answer the following question: \n\n {question}. \n\n we have collected some reasoning paths from a "
              f"knowledge graph to help answer it. \n\n")
    return (header + "\n".join(paths)).strip() + "\n" + POST_STR


def parse_answers(text):
    """Answers after the last "The answer is:" (the template's item 1), up to item 2; markdown removed."""
    text = text.replace('**', '')
    i = text.lower().rfind('the answer is')
    if i < 0:
        return []
    seg = re.split(r'\n\s*2\.\s*I used', text[i + len('the answer is'):], flags=re.I)[0].lstrip(' :').strip()
    parts = re.split(r'\n|;|,(?!\s*(?:D\.C\.|Jr\.|Inc\.))|\band\b', seg)
    out = []
    for p in parts:
        p = re.sub(r'^[\s\-\*\d\.\)]+', '', p).strip().strip('.').strip('*').strip()
        if p and p not in out:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval', required=True)
    ap.add_argument('--dataset', default='webqsp')
    ap.add_argument('--model', default='llama3.1:latest')
    ap.add_argument('--url', default='http://127.0.0.1:11435')
    ap.add_argument('--num_ctx', type=int, default=8192)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tag', default='authors_prompt')
    args = ap.parse_args()

    ret = torch.load(args.retrieval, weights_only=False)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.retrieval)), 'reason')
    os.makedirs(out_dir, exist_ok=True)
    resp_file = f'{out_dir}/{args.dataset}-{args.model.replace(":", "_")}-{args.tag}-responses.jsonl'
    pred_file = f'{out_dir}/{args.dataset}-{args.model.replace(":", "_")}-{args.tag}-predictions.jsonl'
    done = {json.loads(l)['id'] for l in open(resp_file)} if os.path.exists(resp_file) else set()

    # every test question, as the paper evaluates; questions absent from the retrieval get no paths
    import pickle
    test = pickle.load(open(f'{ROOT}/retrieve/data_files/{args.dataset}/processed/test.pkl', 'rb'))
    todo = [s for s in test if s['id'] not in done][:args.limit]
    with open(resp_file, 'a') as f:
        for s in tqdm(todo, desc='reason'):
            paths = (ret.get(s['id']) or {}).get('paths') or []
            body = {'model': args.model, 'stream': False,
                    'messages': [{'role': 'user', 'content': user_text(s['question'], paths)}],
                    'options': {'temperature': 0, 'num_predict': 500, 'repeat_penalty': 1.25,
                                'num_ctx': args.num_ctx, 'seed': 0}}
            r = requests.post(f'{args.url}/api/chat', json=body, timeout=600)
            r.raise_for_status()
            j = r.json()
            f.write(json.dumps({'id': s['id'], 'question': s['question'], 'a_entity': s['a_entity'],
                                'n_paths': len(paths), 'response': j['message']['content'],
                                'prompt_tokens': j.get('prompt_eval_count')}) + '\n')
            f.flush()

    rows = [json.loads(l) for l in open(resp_file)]
    n_noparse = 0
    with open(pred_file, 'w') as f:
        for x in rows:
            ans = parse_answers(x['response'])
            n_noparse += not ans
            f.write(json.dumps({'id': x['id'], 'question': x['question'], 'a_entity': x['a_entity'],
                                'prediction': '\n'.join(f'ans: {a}' for a in ans)}) + '\n')
    sys.path.insert(0, f'{ROOT}/reason'); os.chdir(f'{ROOT}/reason')
    from metrics.evaluate_results import eval_results as eval_orig
    import metrics.evaluate_results_corrected as corr
    print(f'{len(rows)} questions, {n_noparse} without a parsable answer, '
          f'mean prompt tokens {sum(x["prompt_tokens"] or 0 for x in rows) / len(rows):.0f}')
    h = eval_orig(pred_file, cal_f1=True, subset=False)
    # evaluate_results_corrected.match was loosened in this repo (commit 84aae95: any shared character 3-gram counts
    # as a match); the SubgraphRAG original (0db77b5) is substring containment after normalisation. Score with the
    # original, and also report the loosened one so it can be compared with older numbers.
    loose = corr.match
    corr.match = lambda s1, s2: corr.normalize(s2) in corr.normalize(s1)
    c = corr.eval_results(pred_file, cal_f1=True, subset=False)
    corr.match = loose
    cl = corr.eval_results(pred_file, cal_f1=True, subset=False)
    print(f'Hit (orig eval) {h[0]:.2f} | corrected eval, original match: Hit@1 {c[0]:.2f} Macro-F1 {c[1]:.2f} '
          f'| corrected eval, loosened 3-gram match (84aae95): Hit@1 {cl[0]:.2f} Macro-F1 {cl[1]:.2f}')


if __name__ == '__main__':
    main()
