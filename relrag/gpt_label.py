"""GPT-4o rational-path labels through OpenRouter, reproducing the authors' RAPL/REL-RAG labelling for the
train split. OpenRouter dropped every OpenAI `:batch` endpoint on 2026-08-26 (batch submit returns 400
"does not have a :batch endpoint"), so `sync` (standard per-request chat completions, as the authors' own
FindRationalPath.py did) is the working path; `submit`/`poll` are kept for batch-capable models.

  candidates  RAPL/src/dataset/retriever_v2.py + RAPL/MakeTrainData.py: nx.DiGraph over the subgraph; for every
              (topic, answer) pair, forward simple paths of length d..d+2 (cut_off_paths, tolerance 2; only the
              shortest paths when there are >50 of them or d>4); sorted by length with np.argsort, top 10.
              Train samples without a shortest path (triple_scores max_path_length None/0) are skipped
              (RetrieverDataset skip_no_path=True), so `idx` matches the authors' sample_<idx>.txt numbering.
  prompt      the exact chat messages of RAPL/FindRationalPath.py, read verbatim from that file with `ast`;
              model gpt-4o, temperature 0, no max_tokens.
  parsing     RAPL/load_post_processed_dataset.py::_parse_rational_paths, minus its end-of-line anchor (see
              parse_rational_paths); matching to candidate paths as in
              RAPL/src/dataset/PostProcessedDataset.py::_match_candidate_paths.

Deviation: samples whose candidate list is empty are not sent (the authors' script sends a question with no
paths; the outcome is the same — no candidate can be matched).

  python gpt_label.py build  --dataset webqsp --split train
  python gpt_label.py sync   --dataset webqsp --split train --limit 20 --budget 0.5   # pilot
  python gpt_label.py sync   --dataset webqsp --split train --all --budget 15         # the rest (resumable)
  python gpt_label.py labels --dataset webqsp --split train                           # (re)write labels.jsonl
  python gpt_label.py nodes  --dataset webqsp --split test --api deepseek --model deepseek-flash --no_reasoning
                                                                     # labels.jsonl -> line-graph paths(_minlen).jsonl
  # another labeller on the same requests (own output dir), then agreement with the gpt-4o labels:
  python gpt_label.py sync    --api deepseek --model deepseek-flash --no_reasoning --limit 20 --budget 0.05
  python gpt_label.py compare --api deepseek --model deepseek-flash --no_reasoning
  (--api deepseek calls api.deepseek.com with DEEPSEEK_API_KEY; --api openrouter with OPENROUTER_API_KEY)
  (batch-capable models only: submit / poll / list)
The OpenRouter key is read from OPENROUTER_API_KEY or /home/minhnv/Lab/SubgraphRAG/.env and never printed.
"""
import argparse, ast, json, os, pickle, re, sys, time
import networkx as nx
import numpy as np
import requests
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
OUT_ROOT = f'{ROOT}//relrag/data/labels_gpt4o'
API = 'https://openrouter.ai/api/v1/batches'
MODEL = 'openai/gpt-4o'


# ---------------------------------------------------------------- candidates (verbatim logic of the authors)
def get_nx_g(h_id_list, r_id_list, t_id_list):
    nx_g = nx.DiGraph()
    for i in range(len(h_id_list)):
        nx_g.add_edge(h_id_list[i], t_id_list[i], triple_id=i, relation_id=r_id_list[i])
    return nx_g


def cut_off_paths(nx_g, q_entity_id, a_entity_id, tolerance=2):
    try:
        shortest_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        if len(shortest_paths) > 50 or len(shortest_paths[0]) > 5:
            return shortest_paths
    except Exception:
        return []
    if len(shortest_paths) == 0:
        return []
    min_len = len(shortest_paths[0]) - 1
    cutoff = min_len + tolerance
    out = []
    for path in nx.all_simple_paths(nx_g, source=q_entity_id, target=a_entity_id, cutoff=cutoff):
        if min_len <= len(path) - 1 <= min_len + tolerance:
            out.append(path)
    return out


def translate_paths(nx_g, paths, id2entity, id2relation):
    translated, reasoning, distances, triple_ids = [], [], [], []
    for path in paths:
        path_len = len(path) - 1
        distances.append(path_len)
        text, rels, tids = ([id2entity[path[0]]] if path else []), [], []
        for i in range(path_len):
            e = nx_g[path[i]][path[i + 1]]
            text += [id2relation[e['relation_id']], id2entity[path[i + 1]]]
            rels.append(id2relation[e['relation_id']])
            tids.append(e['triple_id'])
        translated.append(' -> '.join(text))
        reasoning.append(rels)
        triple_ids.append(tids)
    return translated, reasoning, distances, triple_ids


def candidates(sample, top_k=10):
    id2entity = dict(enumerate(list(sample['text_entity_list']) + list(sample['non_text_entity_list'])))
    id2relation = dict(enumerate(sample['relation_list']))
    nx_g = get_nx_g(sample['h_id_list'], sample['r_id_list'], sample['t_id_list'])
    all_paths = []
    for q_id in sample['q_entity_id_list']:
        for a_id in sample['a_entity_id_list']:
            all_paths.extend(cut_off_paths(nx_g, q_id, a_id, 2))
    if not all_paths:
        return [], [], [], []
    t, r, d, tid = translate_paths(nx_g, all_paths, id2entity, id2relation)
    order = np.argsort(np.array(d))[:top_k]                     # MakeTrainData.get_sorted_paths
    return [t[i] for i in order], [r[i] for i in order], [d[i] for i in order], [tid[i] for i in order]


# ---------------------------------------------------------------- prompt (verbatim from FindRationalPath.py)
def authors_prompt():
    tree = ast.parse(open(f'{ROOT}/RAPL/FindRationalPath.py').read())
    s = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('first_prompt', 'first_response', 'job_start') \
                and isinstance(node.value, ast.Constant):
            s[node.targets[0].id] = node.value.value
    assert len(s) == 3, s.keys()
    return [{'role': 'user', 'content': s['first_prompt']},
            {'role': 'assistant', 'content': s['first_response']},
            {'role': 'user', 'content': s['job_start']}]


def dict_to_text(question, translated, reasoning):
    text = f"Question: {question}\nThe paths are:\n"
    for i, (path, rp) in enumerate(zip(translated, reasoning), 1):
        text += f"{i}. {path}, the corresponding reasoning path is: {rp}\n\n"
    return text.strip()


# ---------------------------------------------------------------- parsing (verbatim logic of the authors)
def parse_rational_paths(text):
    # The authors' regex ends with `\]\s*$`; current gpt-4o appends the explanation on the same line
    # ("1. ['a', 'b'] - This path ..."), which that anchor rejects, so the end anchor is dropped.
    out, in_section = [], False
    pattern = re.compile(r'^\s*\d+\.\s*\[(.*?)\]')
    for line in text.splitlines():
        if 'The rational paths are:' in line:
            in_section = True
            continue
        if in_section:
            m = pattern.match(line.strip())
            if m:
                out.append(tuple(p.strip() for p in m.group(1).split(',')))
    return out


def match_candidates(gt_tuples, cand_paths):
    def clean(x):
        x = x.strip()
        if (x.startswith("'") and x.endswith("'")) or (x.startswith('"') and x.endswith('"')):
            x = x[1:-1].strip()
        return x
    gt = [tuple(clean(x) for x in t) for t in gt_tuples]
    matched = []
    for idx, cand in enumerate([[clean(x) for x in p] for p in cand_paths]):
        if any(len(cand) == len(g) and all(cand[i].lower().strip() == g[i].lower().strip() for i in range(len(g)))
               for g in gt):
            matched.append(idx)
    return matched


# ---------------------------------------------------------------- state helpers
def out_dir(a):
    d = f'{OUT_ROOT}/{a.dataset}_{a.split}'
    os.makedirs(f'{d}/responses', exist_ok=True)
    return d


def run_dir(a):
    """Responses/labels of one labeller. openai/gpt-4o keeps the build directory (where its labels already are);
    any other model/provider/reasoning setting gets data/labels_<slug>/<dataset>_<split>/."""
    if a.api == 'openrouter' and a.model == MODEL and a.provider == 'openai' and not a.no_reasoning:
        return out_dir(a)
    src = a.provider if a.api == 'openrouter' else 'deepseekapi'
    slug = a.model.split('/')[-1] + f'_{src}' + ('_noreason' if a.no_reasoning else '')
    d = f'{ROOT}//relrag/data/labels_{slug}/{a.dataset}_{a.split}'
    os.makedirs(f'{d}/responses', exist_ok=True)
    return d


def api_key(name='OPENROUTER_API_KEY'):
    k = os.environ.get(name)
    if not k:
        for line in open(f'{ROOT}/.env'):
            if line.startswith(f'{name}='):
                k = line.split('=', 1)[1].strip().strip('"').strip("'")
    assert k, f'{name} not found'
    return k


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---------------------------------------------------------------- subcommands
def build(a):
    d = out_dir(a)
    raw = pickle.load(open(f'{ROOT}/retrieve/data_files/{a.dataset}/processed/{a.split}.pkl', 'rb'))
    ts = torch.load(f'{ROOT}/retrieve/data_files/{a.dataset}/triple_scores/{a.split}.pth', weights_only=False)
    prefix = authors_prompt()
    kept = [s for s in raw if ts[s['id']]['max_path_length'] not in (None, 0)]
    n_empty = 0
    with open(f'{d}/candidates.jsonl', 'w') as fc, open(f'{d}/requests.jsonl', 'w') as fr:
        for idx, s in enumerate(kept):
            t, r, dist, tid = candidates(s)
            fc.write(json.dumps({'idx': idx, 'id': s['id'], 'question': s['question'], 'translated_paths': t,
                                 'reasoning_paths': r, 'distances': [int(x) for x in dist], 'triple_ids': tid}) + '\n')
            if not t:
                n_empty += 1
                continue
            msgs = prefix + [{'role': 'user', 'content': dict_to_text(s['question'], t, r)}]
            fr.write(json.dumps({'custom_id': f'sample_{idx}', 'body': {'messages': msgs, 'temperature': 0}}) + '\n')
    print(json.dumps({'raw': len(raw), 'kept_after_skip_no_path': len(kept), 'no_candidate_not_sent': n_empty,
                      'requests': len(kept) - n_empty}))


def submit(a):
    d = out_dir(a)
    reqs = load_jsonl(f'{d}/requests.jsonl')
    batches = load_jsonl(f'{d}/batches.jsonl')
    taken = {c for b in batches if b.get('final_status') not in ('failed', 'expired', 'cancelled') for c in b['custom_ids']}
    taken |= {f[:-4] for f in os.listdir(f'{d}/responses')}
    todo = [r for r in reqs if r['custom_id'] not in taken]
    if not a.all:
        todo = todo[:a.limit]
    if not todo:
        print('nothing to submit'); return
    print(f'submitting {len(todo)} requests ({len(taken)} already submitted/done, {len(reqs)} total)')
    payload = {'endpoint': '/v1/chat/completions', 'model': MODEL, 'provider': {'only': ['openai']},
               'completion_window': '24h', 'requests': todo}          # key order matters for this API
    r = requests.post(API, headers={'Authorization': f'Bearer {api_key()}', 'Content-Type': 'application/json'},
                      data=json.dumps(payload), timeout=600)
    print(r.status_code, r.text[:500])
    r.raise_for_status()
    bid = r.json()['id']
    with open(f'{d}/batches.jsonl', 'a') as f:
        f.write(json.dumps({'batch_id': bid, 'submitted_at': time.strftime('%F %T'),
                            'custom_ids': [x['custom_id'] for x in todo]}) + '\n')
    print('batch id', bid)


def poll(a):
    d = out_dir(a)
    batches = load_jsonl(f'{d}/batches.jsonl')
    hdr = {'Authorization': f'Bearer {api_key()}'}
    for b in batches:
        if b.get('final_status'):
            continue
        j = requests.get(f"{API}/{b['batch_id']}", headers=hdr, timeout=600).json()
        st = j.get('status')
        print(b['batch_id'], st, j.get('request_counts'), j.get('usage'))
        if st not in ('completed', 'failed', 'expired', 'cancelled'):
            continue
        for res in j.get('results') or []:
            cid = res['custom_id']
            if res.get('error') or not res.get('response') or res['response'].get('status_code') != 200:
                print('  error', cid, json.dumps(res.get('error') or res.get('response'))[:300]); continue
            body = res['response']['body']
            open(f'{d}/responses/{cid}.txt', 'w').write(body['choices'][0]['message']['content'])
            with open(f'{d}/usage.jsonl', 'a') as f:
                f.write(json.dumps({'custom_id': cid, 'batch_id': b['batch_id'], 'model': body.get('model'),
                                    'usage': body.get('usage'),
                                    'cost_usd': (body.get('usage') or {}).get('cost', 0)}) + '\n')
        b['final_status'], b['request_counts'], b['usage'] = st, j.get('request_counts'), j.get('usage')
    with open(f'{d}/batches.jsonl', 'w') as f:
        f.writelines(json.dumps(b) + '\n' for b in batches)
    write_labels(d, d)


def spent_usd(d):
    return sum(json.loads(l).get('cost_usd', 0) for l in open(f'{d}/usage.jsonl')) if os.path.exists(f'{d}/usage.jsonl') else 0.0


def write_labels(d, cand_dir):
    cands = {c['idx']: c for c in load_jsonl(f'{cand_dir}/candidates.jsonl')}
    n_resp = n_parsed = n_matched = 0
    grp = {'single_rel_path': [0, 0], 'multi_rel_path': [0, 0]}   # [answered, no matched candidate]
    with open(f'{d}/labels.jsonl', 'w') as f:
        for fn in sorted(os.listdir(f'{d}/responses'), key=lambda x: int(x[7:-4])):
            idx = int(fn[7:-4]); c = cands[idx]
            text = open(f'{d}/responses/{fn}').read()
            gt = parse_rational_paths(text)
            m = match_candidates(gt, c['reasoning_paths'])
            n_resp += 1; n_parsed += bool(gt); n_matched += bool(m)
            n_unique = len({tuple(p) for p in c['reasoning_paths']})
            g = grp['single_rel_path' if n_unique == 1 else 'multi_rel_path']; g[0] += 1; g[1] += not m
            f.write(json.dumps({'idx': idx, 'id': c['id'], 'rational_paths': gt, 'matched_candidates': m,
                                'n_candidates': len(c['reasoning_paths']), 'n_unique_rel_paths': n_unique}) + '\n')
    cost = spent_usd(d)
    print(json.dumps({'responses': n_resp, 'with_parsed_paths': n_parsed, 'with_matched_candidate': n_matched,
                      'no_match_by_group': {k: f'{v[1]}/{v[0]}' for k, v in grp.items()},
                      'reported_cost_usd': round(cost, 4)}))


def labels(a):
    write_labels(run_dir(a), out_dir(a))


def nodes(a):
    """labels.jsonl -> line-graph label paths for train.py / eval_triples.py: the LLM-matched candidate instances
    (<=10), as node ids of prepare_data.py records, walk-checked; also a *_minlen file keeping only the shortest.
    Same logic as the former merge_labels.py, for one labeller and any split."""
    d = run_dir(a)
    cands = {c['idx']: c for c in load_jsonl(f'{out_dir(a)}/candidates.jsonl')}
    lab = {x['idx']: x for x in load_jsonl(f'{d}/labels.jsonl')}
    pkl = f'{ROOT}//relrag/data/{a.split}.pkl' if a.dataset == 'webqsp' else f'{ROOT}//relrag/data/{a.dataset}/{a.split}.pkl'
    recs = {r['id']: r for r in pickle.load(open(pkl, 'rb'))}
    st = {'with_paths': 0, 'no_paths': 0, 'not_answered': 0, 'missing_triple': 0, 'not_walk': 0}
    rows = []
    for idx in sorted(cands):
        c, l = cands[idx], lab.get(idx)
        if c['id'] not in recs:
            continue
        rec = recs[c['id']]
        node_of = {int(t): k for k, t in enumerate(rec['orig_triple_idx'])}
        paths = []
        st['not_answered'] += l is None
        for ci in (l['matched_candidates'] if l else [])[:10]:
            tids = c['triple_ids'][ci]
            if not all(x in node_of for x in tids):
                st['missing_triple'] += 1; continue
            p = [node_of[x] for x in tids]
            if any(rec['t'][p[i]] != rec['h'][p[i + 1]] for i in range(len(p) - 1)):
                st['not_walk'] += 1; continue
            paths.append(p)
        st['with_paths' if paths else 'no_paths'] += 1
        rows.append({'id': c['id'], 'rational_paths': l['rational_paths'] if l else [], 'paths': paths})
    with open(f'{d}/paths.jsonl', 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    with open(f'{d}/paths_minlen.jsonl', 'w') as f:
        for r in rows:
            m = min((len(p) for p in r['paths']), default=0)
            f.write(json.dumps(dict(r, paths=[p for p in r['paths'] if len(p) == m])) + '\n')
    print(json.dumps(st), '->', f'{d}/paths.jsonl, paths_minlen.jsonl')


# Standard OpenRouter prices of openai/gpt-4o ($/token), used only when a response carries no reported cost.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 2.5e-6, 1.25e-6, 10e-6
# DeepSeek API ($/token) at PEAK rates (api-docs.deepseek.com/quick_start/pricing, checked 2026-10-01): cache hit,
# cache miss, output. Off-peak is half, so this always over-estimates -- the budget guard stays on the safe side.
DEEPSEEK_PRICES = {'deepseek-flash': (0.006e-6, 0.30e-6, 1.20e-6), 'deepseek-v4-pro': (0.044e-6, 1.32e-6, 3.96e-6)}


def sync(a):
    """Per-request chat completions with a hard budget: no new request is started once the recorded spend
    (OpenRouter-reported cost, else a price-table estimate) reaches --budget. Resumable: samples that already
    have responses/<custom_id>.txt are skipped. Any non-retryable HTTP error (e.g. 402 out of credit) stops all
    workers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    d, bd = run_dir(a), out_dir(a)
    if a.api == 'deepseek':
        assert a.model in DEEPSEEK_PRICES, f'no price table for {a.model}; add it to DEEPSEEK_PRICES first'
    reqs = load_jsonl(f'{bd}/requests.jsonl')
    done = {f[:-4] for f in os.listdir(f'{d}/responses')}
    todo = [r for r in reqs if r['custom_id'] not in done]
    if not a.all:
        todo = todo[:a.limit]
    spent0 = spent_usd(d)
    via = a.provider if a.api == 'openrouter' else 'api.deepseek.com'
    print(f'{a.model} via {via} (reasoning {"off" if a.no_reasoning else "provider default"}) -> {d}')
    print(f'{len(todo)} to send ({len(done)} done of {len(reqs)}); spent so far ${spent0:.4f}; budget ${a.budget:.2f}')
    if spent0 >= a.budget:
        print('budget already reached'); return
    if a.api == 'deepseek':
        url, key = 'https://api.deepseek.com/chat/completions', api_key('DEEPSEEK_API_KEY')
    else:
        url, key = 'https://openrouter.ai/api/v1/chat/completions', api_key()
    lock, state = threading.Lock(), {'spent': spent0, 'stop': None, 'n': 0}

    def one(r):
        with lock:
            if state['stop'] or state['spent'] >= a.budget:
                return
        msgs = r['body']['messages']
        if a.merge_user_turns:   # only if the API rejects the authors' two consecutive user turns
            merged = []
            for m in msgs:
                if merged and merged[-1]['role'] == m['role']:
                    merged[-1] = {'role': m['role'], 'content': merged[-1]['content'] + '\n\n' + m['content']}
                else:
                    merged.append(dict(m))
            msgs = merged
        body = {'model': a.model, 'messages': msgs, 'temperature': 0}
        if a.api == 'deepseek':
            if a.no_reasoning:
                body['thinking'] = {'type': 'disabled'}   # thinking is ON by default and then ignores temperature
        else:
            body.update({'provider': {'only': [a.provider], 'allow_fallbacks': False}, 'usage': {'include': True}})
            if a.no_reasoning:
                body['reasoning'] = {'effort': 'none'}   # OpenRouter's documented way to turn reasoning off
        for attempt in range(6):
            try:
                resp = requests.post(url, timeout=300, json=body, headers={'Authorization': f'Bearer {key}'})
            except requests.RequestException as e:
                err = str(e); time.sleep(2 ** attempt * 5); continue
            try:
                j = resp.json()
            except ValueError:
                j = {}
            if resp.status_code == 200 and 'choices' in j:
                break
            err = f'{resp.status_code} {resp.text[:300]}'
            if resp.status_code not in (408, 429, 500, 502, 503, 504):
                with lock:
                    state['stop'] = err
                return
            time.sleep(2 ** attempt * 5)
        else:
            print('  gave up', r['custom_id'], err); return
        u = j.get('usage') or {}
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        if a.api == 'deepseek':          # DeepSeek reports tokens only; cost at peak rates (upper bound)
            hit, miss, out = DEEPSEEK_PRICES[a.model]
            est = u.get('prompt_cache_hit_tokens', 0) * hit + u.get('prompt_cache_miss_tokens', 0) * miss \
                + u.get('completion_tokens', 0) * out
        elif a.model == MODEL:
            est = (u.get('prompt_tokens', 0) - cached) * PRICE_IN + cached * PRICE_CACHED + u.get('completion_tokens', 0) * PRICE_OUT
        else:
            est = 0.0                    # other OpenRouter models rely on the reported cost
        cost = u.get('cost') if u.get('cost') is not None else est
        with lock:
            open(f"{d}/responses/{r['custom_id']}.txt", 'w').write(j['choices'][0]['message']['content'])
            with open(f'{d}/usage.jsonl', 'a') as f:
                f.write(json.dumps({'custom_id': r['custom_id'], 'model': j.get('model'), 'provider': j.get('provider'),
                                    'system_fingerprint': j.get('system_fingerprint'), 'utc': time.strftime('%F %T', time.gmtime()),
                                    'usage': u, 'cost_usd': cost, 'cost_reported': u.get('cost') is not None}) + '\n')
            state['spent'] += cost; state['n'] += 1
            if u.get('cost') is None and a.api == 'openrouter' and a.model != MODEL:
                print('  WARNING: no reported cost for', r['custom_id'], '- budget guard cannot count it')
            if state['n'] % 50 == 0:
                print(f"  {state['n']} done, spent ${state['spent']:.4f}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    if state['stop']:
        print('STOPPED on non-retryable error:', state['stop'])
    print(f"sent {state['n']}, total spent ${state['spent']:.4f}" + (' (DeepSeek: estimate at peak rates, upper bound)' if a.api == 'deepseek' else ''))
    rt = [(x['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for x in load_jsonl(f'{d}/usage.jsonl')]
    if rt:
        print(f'reasoning tokens: mean {sum(rt) / len(rt):.1f} per response (should be 0 with --no_reasoning)')
    hits = [x['usage'].get('prompt_cache_hit_tokens') for x in load_jsonl(f'{d}/usage.jsonl')]
    if hits and hits[0] is not None:
        tot = [x['usage'].get('prompt_tokens', 0) for x in load_jsonl(f'{d}/usage.jsonl')]
        print(f'prompt cache hit: {sum(h or 0 for h in hits) / max(sum(tot), 1) * 100:.0f}% of prompt tokens')
    write_labels(d, bd)


def compare(a):
    """Agreement of this labeller with the gpt-4o labels on the questions both answered (EXP-15 categories), split by
    whether the question offers one or several distinct relation paths. Disjoint cases go to compare_disjoint.jsonl
    for reading by hand."""
    bd, d = out_dir(a), run_dir(a)
    assert d != bd, 'pass the --model/--provider of the labeller to compare with gpt-4o'
    cands = {c['idx']: c for c in load_jsonl(f'{bd}/candidates.jsonl')}
    ref = {x['idx']: x for x in load_jsonl(f'{bd}/labels.jsonl')}
    new = {x['idx']: x for x in load_jsonl(f'{d}/labels.jsonl')}
    from collections import Counter
    st, disjoint = Counter(), []
    for idx in sorted(set(ref) & set(new)):
        paths = [tuple(p) for p in cands[idx]['reasoning_paths']]
        grp = 'multi' if len(set(paths)) > 1 else 'single'
        r = {paths[i] for i in ref[idx]['matched_candidates']}
        n = {paths[i] for i in new[idx]['matched_candidates']}
        if r == n: k = 'identical'
        elif not n: k = 'new_empty'
        elif not r: k = 'gpt4o_empty'
        elif r < n: k = 'new_superset'
        elif n < r: k = 'new_subset'
        elif r & n: k = 'partial_overlap'
        else:
            k = 'disjoint'
            disjoint.append({'idx': idx, 'question': cands[idx]['question'],
                             'gpt4o': sorted(r), 'new': sorted(n)})
        st[(grp, k)] += 1
    for grp in ('multi', 'single'):
        tot = sum(v for (g, _), v in st.items() if g == grp)
        if tot:
            print(f'{grp}-relation-path questions ({tot}):', {k: f'{v} ({v / tot * 100:.1f}%)' for (g, k), v in sorted(st.items()) if g == grp})
    with open(f'{d}/compare_disjoint.jsonl', 'w') as f:
        f.writelines(json.dumps(x) + '\n' for x in disjoint)
    print(f'{len(disjoint)} disjoint cases written to {d}/compare_disjoint.jsonl')


def list_batches(a):
    """Batches known to OpenRouter -- check this before re-running `submit` after a network error, since the
    server may have accepted a batch that was never recorded in batches.jsonl."""
    j = requests.get(API, headers={'Authorization': f'Bearer {api_key()}'}, timeout=600).json()
    for b in j.get('data', j if isinstance(j, list) else []):
        print(b.get('id'), b.get('status'), b.get('created_at'), b.get('request_counts'), (b.get('usage') or {}).get('cost'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'sync', 'labels', 'nodes', 'compare', 'submit', 'poll', 'list'])
    ap.add_argument('--dataset', default='webqsp')
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--budget', type=float, default=0.5, help='sync: stop starting new requests once spend reaches this many USD')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--api', choices=['openrouter', 'deepseek'], default='openrouter')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--provider', default='openai', help='OpenRouter provider slug to pin (no fallbacks)')
    ap.add_argument('--no_reasoning', action='store_true', help="turn thinking off (OpenRouter: reasoning.effort=none; DeepSeek: thinking.type=disabled)")
    ap.add_argument('--merge_user_turns', action='store_true', help='join consecutive same-role messages (deviation; only if the API rejects them)')
    a = ap.parse_args()
    {'build': build, 'sync': sync, 'labels': labels, 'nodes': nodes, 'compare': compare, 'submit': submit, 'poll': poll,
     'list': list_batches}[a.cmd](a)
