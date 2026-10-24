"""Prompt-structure lab (10-17): how should retrieved evidence be presented to the reasoner? Local Llama-3.1-8B Q4 via
ollama (no paid API), dev = VALIDATION splits (dev_<ds>_val.pth from RBF+BGE+WL+SPARQL-v2, top-500 with gold answers
and topic names), scored like the papers (Hit: original SubgraphRAG eval; Macro-F1: corrected eval with the original
substring match). One resumable jsonl per (variant, dataset) under results/prompt_lab/.

Variants (all keep SubgraphRAG's sys_icl_dc frame: system prompt + one ICL example + the question; the ICL example is
converted with the same function as the test context):
  T0  triples "(h,r,t)" in score order (the current pipeline)
  G0  T0's prompt and ICL answer, the evidence regrouped by candidate answer with CVT facts (ctx_candidates)
  T1  triples with short relation names (last segment, '_' -> ' ') and CVT ids replaced by readable placeholders
  P1  reasoning paths from the topic entity: "A -> rel -> B -> rel -> C" (CVT hops folded into the path), best first
  P2  P1 + the list of candidate path ends + instructions to pick the candidates whose relation chain matches the
      question and to use own knowledge only when no path fits
  P2a P1 + the P2 instructions, no candidate list        P2b P1 + the candidate list, no instructions
  P3  P2 + an ICL answer rewritten to demonstrate the procedure (name the relation chain, keep the matching paths,
      discard the others, answer)
  P4  P3 fixed: the ICL answer lines carry only the entity name (P3's "2014 (2014 World Series)" made the model
      annotate and repeat answers), plus a format instruction (each entity once, always end with the "ans:" lines);
      meant for k = 100
  python prompt_lab.py <variant> <webqsp|cwq> [--n 300] [--k 50] [--workers 2]
Test check (--test): the same retriever's top-500 TEST export (TEST_SRC), topics and gold answers from SubgraphRAG's
scored_triples reference files, a fresh sample (seed 1); results go to <variant>_<ds>_k<k>_test.jsonl.
Full run on a bf16 server (--api): any OpenAI-compatible endpoint (vLLM), URL from VLLM_URL, model from VLLM_MODEL;
SubgraphRAG decoding (temperature 0, seed 0, max_tokens 4000); --n 0 = every question; --src <export> replaces the
retriever (top-500 export {id: {'scored_triples'}}, e.g. SubgraphRAG's); results go to results/prompt_lab/<--tag>.jsonl.
Every row stores finish_reason and completion_tokens. --loop_fix: a response cut at max_tokens (greedy decoding fell into a
repetition loop, ~11% of CWQ on vLLM 0.8) is cut where its first sentence repeats, and the model gets one follow-up turn
(LOOP_FIX) asking for the final ans: lines only; the original response is kept as prediction_raw. Same rule for every
retriever and prompt; report it next to the plain numbers since it adds a turn to the protocol.
"""
import argparse, collections, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_results as orig, evaluate_results_corrected as corr
corr.match = orig.match
import prompts

OLLAMA = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.1:latest')
CVT = re.compile(r'^[mg]\.[0-9a-z_]+$')
ICL_TOPIC = ['Lou Seal']
TEST_SRC = {'webqsp': '../retrieve/exp_qrdde/runs/webqsp_rbfnb_bge_wl_sparql2_s42/adr_webqsp_base_mm1_full.pth',
            'cwq': '../retrieve/exp_qrdde/runs/cwq_rbfnb_bge_wl_sparql2_s42/adr_cwq_base_mm1_full.pth'}
TEST_REF = {'webqsp': 'scored_triples/webqsp_240912_unidir_test.pth', 'cwq': 'scored_triples/cwq_240907_unidir_test.pth',
            'grailqa_paper': '/home/minhnv/Lab/SubgraphRAG/data_ext/grailqa_paper/test.pkl'}   # 728 of the paper's 874 questions


def load_reference(dataset):
    """{question id: {'question', 'q_entity', 'a_entity'}} of the test set"""
    if TEST_REF[dataset].endswith('.pkl'):                     # external sets: records of the old ext_data.py
        import pickle
        return {r['id']: {'question': r['question'], 'q_entity': [r['names'][i] for i in r['q_ent']],
                          'a_entity': [r['names'][i] for i in r['a_ent']]} for r in pickle.load(open(TEST_REF[dataset], 'rb'))}
    return torch.load(TEST_REF[dataset], weights_only=False)
DC = ['when', 'what year', 'which year', 'where', 'sport', 'what countr', 'language', 'nba finals', 'world series']


def short(rel):
    return rel.split('.')[-1].replace('_', ' ')


# ----------------------------------------------------------------------------------------------- context builders
def ctx_triples(T, topics, short_names=False):
    names = {}
    def nm(x):
        if short_names and CVT.match(x):
            names.setdefault(x, f'[event {len(names) + 1}]')
            return names[x]
        return x
    return 'Triplets:\n' + '\n'.join(f'({nm(h)},{short(r) if short_names else r},{nm(t)})' for h, r, t in T)


def paths(T, topics, max_hops=3, max_paths=40):
    """simple paths from a topic over the context's triples (both directions); CVT nodes are folded: they can be passed
    through but never end a path; a path is scored by its rank of first appearance (best triples first)"""
    adj = collections.defaultdict(list)
    for k, (h, r, t) in enumerate(T):
        adj[h].append((t, short(r), k)); adj[t].append((h, short(r) + ' (of)', k))
    out, seen = [], set()
    for s in topics:
        if s not in adj:
            continue
        stack = [(s, [s], [], 0)]
        while stack:
            node, nodes, rels, worst = stack.pop()
            if len(rels) and not CVT.match(node):
                key = (tuple(n for n in nodes if not CVT.match(n)), tuple(rels))
                if key not in seen:
                    seen.add(key); out.append((worst, nodes, rels))
            if len(rels) >= max_hops:
                continue
            for v, r, k in adj[node]:
                if v not in nodes:
                    stack.append((v, nodes + [v], rels + [r], max(worst, k)))
    out.sort(key=lambda x: (x[0], len(x[2])))
    lines, ends = [], []
    for _, nodes, rels in out[:max_paths]:
        parts = [nodes[0]]
        for n, r in zip(nodes[1:], rels):
            parts.append(f'-> {r} ->'); parts.append('[...]' if CVT.match(n) else n)
        lines.append(' '.join(parts).replace('-> [...] ->', '->'))
        if nodes[-1] not in ends:
            ends.append(nodes[-1])
    return lines, ends


def ranked_path_lines(ranked, max_paths=40):
    """P2 lines from paths ranked by the retriever itself (context field 'ranked_paths': [(nodes, [(relation, reverse)])],
    best first), formatted and de-duplicated as paths() does"""
    lines, ends, seen = [], [], set()
    for nodes, steps in ranked:
        rels = tuple(short(r) + (' (of)' if rev else '') for r, rev in steps)
        key = (tuple(n for n in nodes if not CVT.match(n)), rels)
        if key in seen:
            continue
        seen.add(key)
        parts = [nodes[0]]
        for n, r in zip(nodes[1:], rels):
            parts.append(f'-> {r} ->'); parts.append('[...]' if CVT.match(n) else n)
        lines.append(' '.join(parts).replace('-> [...] ->', '->'))
        if nodes[-1] not in ends:
            ends.append(nodes[-1])
        if len(lines) == max_paths:
            break
    return lines, ends


def ctx_paths(T, topics, with_cands=False, ranked=None):
    lines, ends = ranked_path_lines(ranked) if ranked else paths(T, topics)
    s = 'Reasoning paths (from the question entity, most relevant first):\n' + '\n'.join(f'{i + 1}. {l}' for i, l in enumerate(lines))
    if with_cands:
        s += '\n\nCandidate answers (path ends): ' + '; '.join(ends[:30])
    return s


G0_MAX_CANDIDATES = 40                                         # --g0_candidates


def ctx_candidates(T, topics, max_hops=3, max_candidates=40):
    """G0 (10-22): the same triples regrouped by candidate answer. Every named entity reached from a question entity by a
    simple path of <= max_hops triples (mediator / CVT nodes only in the middle) is listed once, best first (rank of
    the path's worst triple), with its paths; a CVT on a path is shown with its other facts (role, character, dates...),
    the conditions that tell siblings apart. Triples on no such path follow as plain triplets, so nothing is lost."""
    adj = collections.defaultdict(list)
    for k, (h, r, t) in enumerate(T):
        adj[h].append((t, r, k, False)); adj[t].append((h, r, k, True))
    found = {}                                                  # end -> [(worst rank, nodes, steps)]
    for s in topics:
        stack = [(s, [s], [], -1)]
        while stack:
            node, nodes, steps, worst = stack.pop()
            if steps and not CVT.match(node) and node not in topics:
                found.setdefault(node, []).append((worst, nodes, steps))
            if len(steps) >= max_hops:
                continue
            for v, r, k, rev in adj[node]:
                if v not in nodes:
                    stack.append((v, nodes + [v], steps + [(r, k, rev)], max(worst, k)))
    used = set()
    def cvt_facts(c, path_nodes):
        """the CVT's own attributes: not the path itself, not the question entity, no Freebase bookkeeping"""
        facts = {}
        for o, r, k, _ in sorted(adj[c], key=lambda e: e[2]):
            used.add(k)
            if o in path_nodes or o in topics or CVT.match(o) or r.startswith('freebase.'):
                continue
            facts.setdefault((r, o), k)
        return '; '.join(f'{r} = {o}' for r, o in facts)
    ranked = sorted(found.items(), key=lambda kv: min(p[0] for p in kv[1]))[:max_candidates]
    lines = ['Candidate answers (entities reached from the question entity, most relevant first), each with its evidence:']
    for i, (end, ps) in enumerate(ranked, 1):
        lines.append(f'{i}. {end}')
        shown_chains = set()
        for worst, nodes, steps in sorted(ps, key=lambda p: p[0]):
            chain = tuple(r for r, _, _ in steps)
            if chain in shown_chains or len(shown_chains) == 2:          # at most 2 paths, with different relations
                continue
            shown_chains.add(chain); used.update(k for _, k, _ in steps)
            parts = [nodes[0]]
            for n, (r, k, rev) in zip(nodes[1:], steps):
                parts.append(f'-> {r}{" (reverse)" if rev else ""} ->')
                if CVT.match(n):
                    facts = cvt_facts(n, set(nodes)) if len(shown_chains) == 1 else ''   # facts once per candidate
                    parts.append(f'[event{": " + facts if facts else ""}]')
                else:
                    parts.append(n)
            lines.append('   ' + ' '.join(parts))
    rest = [T[k] for k in range(len(T)) if k not in used]
    if rest:
        lines.append('Other facts:\n' + '\n'.join(f'({h},{r},{t})' for h, r, t in rest))
    return '\n'.join(lines)


P2_SYS = (prompts.icl_sys_prompt + ' The evidence is given as reasoning paths that start at the entity of the question.'
          ' First decide which chain of relations the question asks for, then answer with the entities at the end of the'
          ' paths that follow that chain and satisfy every condition of the question (time, type, role, location).'
          ' Several answers are allowed.')


def icl_triples():
    block = prompts.icl_user_prompt.split('Triplets:\n', 1)[1].split('\n\n\nQuestion:', 1)[0]
    return [tuple(l.strip()[1:-1].split(',', 2)) for l in block.strip().split('\n') if l.strip()]


def build(variant, T, topics, question, ranked=None):
    if variant == 'T0':
        c = ctx_triples(T, topics)
    elif variant == 'G0':                                      # T0 prompt, evidence grouped by candidate answer
        c = ctx_candidates(T, topics, max_candidates=G0_MAX_CANDIDATES)
    elif variant == 'T1':
        c = ctx_triples(T, topics, short_names=True)
    else:
        c = ctx_paths(T, topics, with_cands=variant in ('P2'), ranked=ranked)
    q = question if question.endswith('?') else question + '?'
    return c + '\n\n\nQuestion:\n' + q


def messages(variant, T, topics, question, ranked=None):
    sys_p = P2_SYS if variant == "P2" else prompts.icl_sys_prompt
    icl_q = prompts.icl_user_prompt.split('Question:\n', 1)[1]
    return [{'role': 'system', 'content': sys_p},
            {'role': 'user', 'content': build(variant, icl_triples(), ICL_TOPIC, icl_q)},
            {'role': 'assistant', 'content': prompts.icl_ass_prompt.replace('the triplets', 'the paths') if variant.startswith('P') else prompts.icl_ass_prompt},
            {'role': 'user', 'content': build(variant, T, topics, question, ranked)}]


LOOP_FIX = ('Your previous answer was cut off because it started repeating itself. Based on the evidence and your reasoning '
            'above, give only the final answers, one per line, each prefixed with "ans:".')
NO_ANS_FIX = ('Your previous answer did not end with any line prefixed "ans:". Based on the evidence and your reasoning '
              'above, give only the final answers, one per line, each prefixed with "ans:".')
MAX_CLEAN_CHARS = 6000


def broken(pred, fin):
    """pure reasoning quality check (repair_runs.is_broken, inlined to avoid a circular import): no 'ans:' line,
    a looping response, or cut at max_tokens"""
    return 'ans:' not in pred or len(pred) > MAX_CLEAN_CHARS or fin == 'length'


def cut_loop(text, min_len=20):
    """prefix of a looping response up to the first repeated sentence"""
    seen, pos = set(), 0
    for m in re.finditer(r'[^\n.]+(?:[\n.]|$)', text):
        key = ' '.join(m.group().split()).lower()
        if len(key) >= min_len:
            if key in seen:
                return text[:m.start()].rstrip()
            seen.add(key)
        pos = m.end()
    return text


def ask(msgs, api=False, max_tokens=4000):
    """-> (content, finish_reason, completion_tokens)"""
    if api:
        for attempt in range(6):                               # transient gateway errors under heavy concurrency
            try:
                r = requests.post(f"{os.environ['VLLM_URL'].rstrip('/')}/chat/completions", timeout=1200,
                                  headers={'Authorization': f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"},
                                  json={'model': os.environ['VLLM_MODEL'], 'messages': msgs, 'temperature': 0, 'seed': 0, 'max_tokens': max_tokens})
                r.raise_for_status()
                j = r.json()
                return j['choices'][0]['message']['content'], j['choices'][0].get('finish_reason'), j.get('usage', {}).get('completion_tokens')
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))
    r = requests.post(f'{OLLAMA}/api/chat', json={'model': MODEL, 'messages': msgs, 'stream': False,
                                                  'options': {'temperature': 0, 'seed': 0, 'num_ctx': 8192, 'num_predict': min(max_tokens, 512)}}, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j['message']['content'], j.get('done_reason'), j.get('eval_count')


# ------------------------------------------------------------------------------------------------------- scoring
def score(pred, gold, question):
    dc = any(k in question.lower() for k in DC)
    hit = bool(orig.eval_hit(pred, gold, dc))
    p = corr.get_pred(pred, None)
    ans = sorted(set(gold), key=len, reverse=True)
    prec = corr.eval_precision(p, ans, dc)[0]; rec = corr.eval_recall(p, ans, dc)[0]
    return hit, corr.eval_f1(prec, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['T0', 'T1', 'G0', 'P1', 'P2']); ap.add_argument('dataset', choices=['webqsp', 'cwq', 'grailqa_paper'])
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--k', type=int, default=50); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--test', action='store_true'); ap.add_argument('--api', action='store_true')
    ap.add_argument('--src'); ap.add_argument('--tag'); ap.add_argument('--loop_fix', action='store_true')
    ap.add_argument('--loop_fix_max', type=int, default=1,
                    help='--loop_fix: retry a broken response (cut at max_tokens, looping, or no "ans:" line) this '
                         'many times (each retry continues the SAME conversation, asking again for just the ans: '
                         'lines); stops early once the response is clean. No chain/answer substitution (pure reasoning).')
    ap.add_argument('--ids', help='file with one question id per line: only these questions (e.g. the errors of a run)')
    ap.add_argument('--g0_candidates', type=int, default=40, help='G0: at most this many candidate answers are listed')
    a = ap.parse_args()
    global G0_MAX_CANDIDATES
    G0_MAX_CANDIDATES = a.g0_candidates
    if a.test:
        src, ref = torch.load(a.src or TEST_SRC[a.dataset], weights_only=False), load_reference(a.dataset)
        dev = {i: {'question': r['question'], 'a_entity': r.get('a_entity', []), 'q_entity': r['q_entity'],
                   'scored_triples': src[i]['scored_triples'], 'ranked_paths': src[i].get('ranked_paths')} for i, r in ref.items() if i in src}   # transfer exports: subset
    else:                                                      # --src here: another val context file of the same shape
        dev = torch.load(a.src or f'custom_retrieval/dev_{a.dataset}_val.pth', weights_only=False)
    ids = sorted(i for i, v in dev.items() if v['a_entity'])
    random.Random(1 if a.test else 0).shuffle(ids); ids = ids[:a.n] if a.n else sorted(ids)
    if a.ids:
        wanted = set(open(a.ids).read().split()); ids = [i for i in ids if i in wanted]
    out_p = f'results/prompt_lab/{a.tag or f"{a.variant}_{a.dataset}_k{a.k}" + ("_test" if a.test else "")}.jsonl'
    os.makedirs('results/prompt_lab', exist_ok=True)
    done = {json.loads(l)['id'] for l in open(out_p)} if os.path.exists(out_p) else set()
    lock = threading.Lock()
    def run(i):
        v = dev[i]; T = [tuple(t[:3]) for t in v['scored_triples'][:a.k]]
        msgs = messages(a.variant, T, v['q_entity'], v['question'], v.get('ranked_paths'))
        pred, fin, ntok = ask(msgs, a.api)
        row = {'id': i, 'question': v['question'], 'gold': v['a_entity'], 'finish_reason': fin, 'completion_tokens': ntok}
        attempts = []
        if a.loop_fix:
            convo = msgs
            for _ in range(a.loop_fix_max):
                if not broken(pred, fin):
                    break
                attempts.append(pred)
                shown = cut_loop(pred) if fin == 'length' else pred            # what the model "said" before retrying
                msg = LOOP_FIX if fin == 'length' else NO_ANS_FIX
                convo = convo + [{'role': 'assistant', 'content': shown}, {'role': 'user', 'content': msg}]
                tail, fin, _ = ask(convo, a.api, 256)
                pred = shown + '\n\n' + tail
        if attempts:
            row['prediction_raw'] = attempts[0]; row['loop_fix_attempts'] = len(attempts)
        hit, f1 = score(pred, v['a_entity'], v['question'])
        row.update({'hit': hit, 'f1': f1, 'prediction': pred, 'prompt_chars': sum(len(m['content']) for m in msgs)})
        with lock, open(out_p, 'a') as f:
            f.write(json.dumps(row) + '\n')
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, [i for i in ids if i not in done]))
    R = [json.loads(l) for l in open(out_p)]
    R = [r for r in R if r['id'] in set(ids)]
    print(f"{a.variant} {a.dataset} k={a.k}: n={len(R)} Hit {100 * sum(r['hit'] for r in R) / len(R):.1f} "
          f"Macro-F1 {100 * sum(r['f1'] for r in R) / len(R):.1f} prompt chars {sum(r['prompt_chars'] for r in R) / len(R):.0f} "
          f"cut at max_tokens {sum(r.get('finish_reason') == 'length' for r in R)}")


if __name__ == '__main__':
    main()
