"""Prompt-structure lab (10-17): how should retrieved evidence be presented to the reasoner? Local Llama-3.1-8B Q4 via
ollama (no paid API), dev = VALIDATION splits (dev_<ds>_val.pth from RBF+BGE+WL+SPARQL-v2, top-500 with gold answers
and topic names), scored like the papers (Hit: original SubgraphRAG eval; Macro-F1: corrected eval with the original
substring match). One resumable jsonl per (variant, dataset) under results/prompt_lab/.

Variants (all keep SubgraphRAG's sys_icl_dc frame: system prompt + one ICL example + the question; the ICL example is
converted with the same function as the test context):
  T0  triples "(h,r,t)" in score order (the current pipeline)
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
TEST_REF = {'webqsp': 'scored_triples/webqsp_240912_unidir_test.pth', 'cwq': 'scored_triples/cwq_240907_unidir_test.pth'}
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


def ctx_paths(T, topics, with_cands=False):
    lines, ends = paths(T, topics)
    s = 'Reasoning paths (from the question entity, most relevant first):\n' + '\n'.join(f'{i + 1}. {l}' for i, l in enumerate(lines))
    if with_cands:
        s += '\n\nCandidate answers (path ends): ' + '; '.join(ends[:30])
    return s


P2_SYS = (prompts.icl_sys_prompt + ' The evidence is given as reasoning paths that start at the entity of the question.'
          ' First decide which chain of relations the question asks for, then answer with the entities at the end of the'
          ' paths that follow that chain and satisfy every condition of the question (time, type, role, location).'
          ' Several answers are allowed. Only if no path fits, answer from your own knowledge.')


P3_ICL_ANS = """The question asks for the years in which the team whose mascot is Lou Seal won the World Series, so the chain of relations is: Lou Seal -> team -> (a team) -> championships -> (a World Series).

Paths that follow this chain:
- Lou Seal -> team -> San Francisco Giants -> championships -> 2010 World Series
- Lou Seal -> team -> San Francisco Giants -> championships -> 2012 World Series
- Lou Seal -> team -> San Francisco Giants -> championships -> 2014 World Series

Paths with other relations (event, mascot, owner, stadium) do not answer the question, and every World Series reached through "championships" is a win, so all three are answers.

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""


P4_SYS = (P2_SYS + ' Write each answer entity exactly once, using its name as it appears in the paths, without any'
          ' explanation on the answer lines, and always finish with the answer lines.')
P4_ICL_ANS = P3_ICL_ANS.replace('ans: 2014 (2014 World Series)\nans: 2012 (2012 World Series)\nans: 2010 (2010 World Series)',
                                'ans: 2014 World Series\nans: 2012 World Series\nans: 2010 World Series')


def icl_triples():
    block = prompts.icl_user_prompt.split('Triplets:\n', 1)[1].split('\n\n\nQuestion:', 1)[0]
    return [tuple(l.strip()[1:-1].split(',', 2)) for l in block.strip().split('\n') if l.strip()]


def build(variant, T, topics, question):
    if variant == 'T0':
        c = ctx_triples(T, topics)
    elif variant == 'T1':
        c = ctx_triples(T, topics, short_names=True)
    else:
        c = ctx_paths(T, topics, with_cands=variant in ('P2', 'P2b', 'P3', 'P4'))
    q = question if question.endswith('?') else question + '?'
    return c + '\n\n\nQuestion:\n' + q


def messages(variant, T, topics, question):
    sys_p = P4_SYS if variant == 'P4' else P2_SYS if variant in ('P2', 'P2a', 'P3') else prompts.icl_sys_prompt
    icl_q = prompts.icl_user_prompt.split('Question:\n', 1)[1]
    return [{'role': 'system', 'content': sys_p},
            {'role': 'user', 'content': build(variant, icl_triples(), ICL_TOPIC, icl_q)},
            {'role': 'assistant', 'content': P4_ICL_ANS if variant == 'P4' else P3_ICL_ANS if variant == 'P3' else
             prompts.icl_ass_prompt.replace('the triplets', 'the paths') if variant.startswith('P') else prompts.icl_ass_prompt},
            {'role': 'user', 'content': build(variant, T, topics, question)}]


LOOP_FIX = ('Your previous answer was cut off because it started repeating itself. Based on the evidence and your reasoning '
            'above, give only the final answers, one per line, each prefixed with "ans:".')


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
    ap.add_argument('variant', choices=['T0', 'T1', 'P1', 'P2', 'P2a', 'P2b', 'P3', 'P4']); ap.add_argument('dataset', choices=['webqsp', 'cwq'])
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--k', type=int, default=50); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--test', action='store_true'); ap.add_argument('--api', action='store_true')
    ap.add_argument('--src'); ap.add_argument('--tag'); ap.add_argument('--loop_fix', action='store_true')
    a = ap.parse_args()
    if a.test:
        src, ref = torch.load(a.src or TEST_SRC[a.dataset], weights_only=False), torch.load(TEST_REF[a.dataset], weights_only=False)
        dev = {i: {'question': r['question'], 'a_entity': r.get('a_entity', []), 'q_entity': r['q_entity'],
                   'scored_triples': src[i]['scored_triples']} for i, r in ref.items() if i in src}   # transfer exports: subset
    else:                                                      # --src here: another val context file of the same shape
        dev = torch.load(a.src or f'custom_retrieval/dev_{a.dataset}_val.pth', weights_only=False)
    ids = sorted(i for i, v in dev.items() if v['a_entity'])
    random.Random(1 if a.test else 0).shuffle(ids); ids = ids[:a.n] if a.n else sorted(ids)
    out_p = f'results/prompt_lab/{a.tag or f"{a.variant}_{a.dataset}_k{a.k}" + ("_test" if a.test else "")}.jsonl'
    os.makedirs('results/prompt_lab', exist_ok=True)
    done = {json.loads(l)['id'] for l in open(out_p)} if os.path.exists(out_p) else set()
    lock = threading.Lock()
    def run(i):
        v = dev[i]; T = [tuple(t[:3]) for t in v['scored_triples'][:a.k]]
        msgs = messages(a.variant, T, v['q_entity'], v['question'])
        pred, fin, ntok = ask(msgs, a.api)
        row = {'id': i, 'question': v['question'], 'gold': v['a_entity'], 'finish_reason': fin, 'completion_tokens': ntok}
        if a.loop_fix and fin == 'length':
            head = cut_loop(pred)
            tail, _, _ = ask(msgs + [{'role': 'assistant', 'content': head}, {'role': 'user', 'content': LOOP_FIX}], a.api, 256)
            row['prediction_raw'] = pred; pred = head + '\n\n' + tail
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
