"""Step 6 (OpenRouter) — the authors' answer generation (RAPL/generate_answers_llama31_8b.py) with Llama-3.1-8B-Instruct
served through OpenRouter, provider pinned to DeepInfra (fp8, 128k context, no fallback). Same as
generate_answers_ollama.py in everything that matters for the result:
  - prompt = sample text .strip() + "\\n" + the authors' post_str (read verbatim with ast), one user turn, the HF
    Llama-3.1 chat template with its default system block, sent raw to the /completions endpoint;
  - saved file = what the authors save (prompt + completion, special tokens dropped), in llama31_8b_generated_answers/.
Generation: max_tokens 500, temperature 0, top_p 1, repetition_penalty 1.25, seed 1 (authors: beam 5, bf16 — not
available). Cost read from the per-request usage; the run stops before exceeding --budget (USD). Resumable.
Key: OPENROUTER_API_KEY from the project .env (never printed).
  python rapl_faithful/generate_answers_openrouter.py --file_dir <dir of sample_*.txt> --limit 20 --budget 0.05
"""
import argparse, concurrent.futures as cf, json, os, re, sys, threading, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_answers_ollama import authors_post_str, HF_SYSTEM

ENV = os.environ.get('SGRAG_ROOT', '/home/minhnv/Lab/SubgraphRAG') + '/.env'
URL = 'https://openrouter.ai/api/v1/completions'


def key():
    for line in open(ENV):
        m = re.match(r'\s*OPENROUTER_API_KEY\s*=\s*"?([^"\s]+)"?', line)
        if m:
            return m.group(1)
    raise RuntimeError('OPENROUTER_API_KEY not in .env')


def post(payload, k, tries=6):
    for a in range(tries):
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={'Authorization': f'Bearer {k}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            if 'choices' in d and d['choices']:
                return d
            err = d.get('error')
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: {e.read()[:200]!r}'
            if e.code in (400, 401, 402, 403):
                raise RuntimeError(err)
        except Exception as e:                                   # network / timeout
            err = repr(e)
        time.sleep(min(60, 2 ** a * 2))
    raise RuntimeError(f'gave up after {tries} tries: {err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_dir', required=True)
    ap.add_argument('--model', default='meta-llama/llama-3.1-8b-instruct')
    ap.add_argument('--provider', default='DeepInfra')
    ap.add_argument('--quant', default='fp8')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='first N files only (pilot)')
    ap.add_argument('--budget', type=float, required=True, help='hard stop, USD, for this invocation')
    args = ap.parse_args()
    k, post_str = key(), authors_post_str()
    files = sorted((f for f in os.listdir(args.file_dir) if f.startswith('sample_') and f.endswith('.txt')),
                   key=lambda f: int(f[7:-4]))
    if args.limit:
        files = files[:args.limit]
    out_dir = os.path.join(args.file_dir, 'llama31_8b_generated_answers')
    os.makedirs(out_dir, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(out_dir, f))]
    print(f'{len(files)} sample files, {len(files) - len(todo)} already answered, {len(todo)} to do; budget ${args.budget}', flush=True)
    log = open(os.path.join(out_dir, '_openrouter_log.jsonl'), 'a')
    lock, state = threading.Lock(), {'cost': 0.0, 'stop': False, 'done': 0, 'in': 0, 'out': 0}

    def one(fname):
        if state['stop']:
            return None
        content = (open(os.path.join(args.file_dir, fname), encoding='utf-8').read().strip() + '\n' + post_str).strip()
        prompt = (f'<|start_header_id|>system<|end_header_id|>\n\n{HF_SYSTEM}<|eot_id|>'
                  f'<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
                  f'<|start_header_id|>assistant<|end_header_id|>\n\n')
        t0 = time.time()
        d = post({'model': args.model, 'prompt': prompt, 'max_tokens': 500, 'temperature': 0, 'top_p': 1,
                  'repetition_penalty': 1.25, 'seed': 1, 'usage': {'include': True},
                  'provider': {'order': [args.provider], 'allow_fallbacks': False, 'quantizations': [args.quant]}}, k)
        completion = d['choices'][0].get('text', '')
        saved = (f'system\n\n{HF_SYSTEM}user\n\n{content}assistant\n\n{completion}').strip()
        with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
            f.write(saved)
        u = d.get('usage', {})
        rec = {'file': fname, 'sec': round(time.time() - t0, 2), 'prompt_tokens': u.get('prompt_tokens'),
               'gen_tokens': u.get('completion_tokens'), 'cost': u.get('cost'), 'provider': d.get('provider'),
               'finish': d['choices'][0].get('finish_reason')}
        with lock:
            state['cost'] += float(u.get('cost') or 0); state['done'] += 1
            state['in'] += u.get('prompt_tokens') or 0; state['out'] += u.get('completion_tokens') or 0
            if state['cost'] >= args.budget * 0.98:
                state['stop'] = True
            log.write(json.dumps(rec) + '\n'); log.flush()
            if state['done'] % 50 == 0 or state['done'] == len(todo):
                print(f"{state['done']}/{len(todo)}  ${state['cost']:.4f}  in {state['in']} out {state['out']} tokens", flush=True)
        return rec

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    print(f"finished: {state['done']} answered, cost ${state['cost']:.4f}, input {state['in']} / output {state['out']} tokens"
          + (' -- STOPPED AT BUDGET' if state['stop'] else ''), flush=True)


if __name__ == '__main__':
    main()
