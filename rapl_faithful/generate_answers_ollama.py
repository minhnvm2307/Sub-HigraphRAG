"""Step 6 — the authors' answer generation (RAPL/generate_answers_llama31_8b.py), with the LLM served by ollama on svuet
instead of local HF transformers. Everything the authors' script fixes is kept:
  - input  = each sample_<i>.txt written by model_inference.py, read and .strip()ped;
  - prompt = that text + "\\n" + post_str, post_str read VERBATIM from the authors' file (ast), as one user message;
  - chat template = HF Llama-3.1 apply_chat_template(add_generation_prompt=True) with no system message, i.e. the default
    system block "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" and the message content trimmed;
    sent to ollama in raw mode so ollama's own template does not change it (--mode chat = ollama's template instead);
  - output = what the authors save: tokenizer.batch_decode(generate(...), skip_special_tokens=True)[0].strip(), and HF
    generate() returns prompt + completion for a decoder-only model, so the saved text CONTAINS THE PROMPT (the
    retrieved paths) followed by the answer. Saved to <file_dir>/llama31_8b_generated_answers/sample_<i>.txt.
Unavoidable differences (ollama): no beam search (authors: num_beams=5, length_penalty=1.2) -> greedy (temperature 0);
model = llama3.1:8b Q4_K_M (authors: bf16). Kept: max_new_tokens 500 -> num_predict 500, repetition_penalty 1.25 ->
repeat_penalty 1.25 over the whole context (repeat_last_n -1, as HF), top_p 1.0, fixed seed.
Speed (shared ollama service on svuet, model already resident): one fixed num_ctx for every request (a different num_ctx
reloads the model), keep_alive -1 (never unloaded between requests), sequential or a few workers, resumable.
  ssh -N -L 21434:localhost:11434 -p 8282 svuet@118.70.181.146 &     # tunnel, then:
  python rapl_faithful/generate_answers_ollama.py --file_dir <model_inference output dir> --url http://localhost:21434
"""
import argparse, ast, concurrent.futures as cf, json, os, time, urllib.request

AUTHORS_FILE = os.environ.get('SGRAG_ROOT', '/home/minhnv/Lab/SubgraphRAG') + '/RAPL/generate_answers_llama31_8b.py'
HF_SYSTEM = 'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n'


def authors_post_str():
    for node in ast.walk(ast.parse(open(AUTHORS_FILE).read())):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'post_str' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('post_str not found in the authors\' script')


def call(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_dir', required=True, help='directory with sample_*.txt from RAPL/model_inference.py')
    ap.add_argument('--url', default='http://localhost:21434', help='ollama on svuet, through an SSH tunnel')
    ap.add_argument('--model', default='llama3.1:8b')
    ap.add_argument('--mode', default='raw', choices=['raw', 'chat'], help='raw = HF Llama-3.1 template (faithful)')
    ap.add_argument('--num_ctx', type=int, default=8192, help='fixed for every request (changing it reloads the model)')
    ap.add_argument('--workers', type=int, default=1, help='parallel requests (useful only if the server runs OLLAMA_NUM_PARALLEL>1)')
    ap.add_argument('--limit', type=int, default=0, help='first N files only (pilot)')
    args = ap.parse_args()

    post_str = authors_post_str()
    files = sorted(f for f in os.listdir(args.file_dir) if f.startswith('sample_') and f.endswith('.txt'))
    files = sorted(files, key=lambda f: int(f[7:-4]))
    if args.limit:
        files = files[:args.limit]
    out_dir = os.path.join(args.file_dir, 'llama31_8b_generated_answers')
    os.makedirs(out_dir, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(out_dir, f))]
    print(f'{len(files)} sample files, {len(files) - len(todo)} already answered, {len(todo)} to do', flush=True)
    opts = {'temperature': 0, 'top_p': 1.0, 'repeat_penalty': 1.25, 'repeat_last_n': -1, 'num_predict': 500,
            'num_ctx': args.num_ctx, 'seed': 1}
    call(f'{args.url}/api/generate', {'model': args.model, 'prompt': '', 'keep_alive': -1, 'options': {'num_ctx': args.num_ctx}})
    log = open(os.path.join(out_dir, '_ollama_log.jsonl'), 'a')

    def one(fname):
        user_text = open(os.path.join(args.file_dir, fname), encoding='utf-8').read().strip()
        content = (user_text + '\n' + post_str).strip()            # HF Llama-3.1 template trims message content
        t0 = time.time()
        if args.mode == 'raw':
            prompt = (f'<|start_header_id|>system<|end_header_id|>\n\n{HF_SYSTEM}<|eot_id|>'
                      f'<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
                      f'<|start_header_id|>assistant<|end_header_id|>\n\n')
            r = call(f'{args.url}/api/generate', {'model': args.model, 'prompt': prompt, 'raw': True, 'stream': False,
                                                  'keep_alive': -1, 'options': opts})
            completion = r['response']
        else:
            r = call(f'{args.url}/api/chat', {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
                                              'stream': False, 'keep_alive': -1, 'options': opts})
            completion = r['message']['content']
        # what the authors save: decode(prompt + completion, skip_special_tokens=True).strip()
        saved = (f'system\n\n{HF_SYSTEM}user\n\n{content}assistant\n\n{completion}').strip()
        with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
            f.write(saved)
        n_in, n_out = r.get('prompt_eval_count', 0), r.get('eval_count', 0)
        return {'file': fname, 'sec': round(time.time() - t0, 2), 'prompt_tokens': n_in, 'gen_tokens': n_out,
                'ctx_risk': n_in + 500 > args.num_ctx}

    t_start, done = time.time(), 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for rec in ex.map(one, todo):
            done += 1
            log.write(json.dumps(rec) + '\n'); log.flush()
            if rec['ctx_risk']:
                print(f"WARNING {rec['file']}: prompt {rec['prompt_tokens']} tokens + 500 > num_ctx {args.num_ctx}", flush=True)
            if done % 20 == 0 or done == len(todo):
                el = time.time() - t_start
                print(f'{done}/{len(todo)}  {el / done:.1f} s/sample  ETA {el / done * (len(todo) - done) / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
