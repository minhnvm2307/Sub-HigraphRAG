"""Why is QA low when the answer is in the context? Per-question error anatomy of one prompt_lab run.
Buckets (exclusive, in this order):
  HIT              paper-style Hit
  NOT_IN_GRAPH     no gold answer in the subgraph at all (data ceiling)
  NOT_RETRIEVED    gold in the subgraph but not among the head/tail entities of the top-k context
  -- gold answer IS in the context, the LLM still misses:
  NO_ANSWER        no ans: line, or the ans: lines say there is no answer
  NEAR_MISS        a predicted answer shares >= 50% tokens with a gold answer (alias / surface form; scorer artefact)
  CHAIN_SIBLING    a predicted answer is an end node of the GOLD relation chain that is not a gold answer (right chain,
                   unresolved constraint: time / order / type / second anchor)
  OTHER_IN_CTX     a predicted answer is another entity of the context (wrong chain or wrong hop)
  OUT_OF_CTX       predicted answers are not context entities (own knowledge / hallucination)
Cross-tabs: literal constraint in the gold query (time FILTER, ORDER BY, comparison), CWQ compositionality type,
number of gold answers. usage: qa_anatomy.py <run.jsonl> <webqsp|cwq> <export.pth> [k]"""
import collections, json, pickle, re, sys
import torch

run_p, ds, src_p = sys.argv[1:4]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 100
ROOT = '/home/minhnv/Lab/SubgraphRAG'
sys.path.insert(0, f'{ROOT}/relrag'); sys.path.insert(0, f'{ROOT}/retrieve/exp_qrdde')
from labels import cwq_gold_queries, triple_lookup, walk_one_trail_per_end, webqsp_gold_queries

REF = {'webqsp': f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth', 'cwq': f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth'}
ref = torch.load(REF[ds], weights_only=False)
src = torch.load(src_p, weights_only=False) if src_p != 'REF' else ref
rows = {r['id']: r for r in map(json.loads, open(run_p))}

# gold relation chains -> end nodes reachable along the chain (gold or not)
recs = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/test.pkl', 'rb'))}
wq, m2n = webqsp_gold_queries()
labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else cwq_gold_queries(recs, m2n)
ends = {}
for qid, lab in labels.items():
    rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
    trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
    front = walk_one_trail_per_end(lab, *triple_lookup(trip))
    ends[qid] = set(front)

# literal constraints in the gold query
lit, ctype = {}, {}
if ds == 'cwq':
    for s in ('test',):
        for q in json.load(open(f'{ROOT}/relrag/data/cwq_orig/ComplexWebQuestions_{s}.json')):
            sp = q['sparql']
            lit[q['ID']] = bool(re.search(r'ORDER BY|xsd:datetime|FILTER\s*\(\s*\?\w+\s*[<>]|FILTER\s*\(\s*xsd', sp))
            ctype[q['ID']] = q['compositionality_type']
else:
    for qid, lab in wq.items():
        lit[qid] = any(c['kind'] in ('literal', 'order_key') for c in lab['constraints'])
    for split in ('test',):
        for q in json.load(open(f'{ROOT}/relrag/data/webqsp_orig/WebQSP/data/WebQSP.{split}.json'))['Questions']:
            p = next((p for p in q.get('Parses', []) if p.get('InferentialChain')), None)
            if p and p.get('Constraints'):
                if any(c.get('ArgumentType') == 'Value' for c in p['Constraints']):
                    lit[q['QuestionId']] = True
            ctype[q['QuestionId']] = f"{len(p['InferentialChain']) if p else 0}-hop"

norm = lambda s: ' '.join(re.sub(r'[^0-9a-z ]', ' ', s.lower()).split())
NONE = re.compile(r'no (answer|information|path)|not (mentioned|provided|available|found)|cannot be determined|unknown|none', re.I)


def bucket(i, r):
    if r['hit']:
        return 'HIT'
    gold = ref[i].get('a_entity') or []
    if not ref[i].get('a_entity_in_graph'):
        return 'NOT_IN_GRAPH'
    ctx = src[i]['scored_triples'][:K] if i in src else []
    ents = {x[0] for x in ctx} | {x[2] for x in ctx}
    if not set(ref[i]['a_entity_in_graph']) & ents:
        return 'NOT_RETRIEVED'
    preds = [p.strip() for p in re.findall(r'ans:\s*(.*)', r['prediction']) if p.strip()]
    if not preds or all(NONE.search(p) for p in preds):
        return 'NO_ANSWER'
    gtok = [set(norm(g).split()) for g in gold]
    for p in preds:
        pt = set(norm(p).split())
        if pt and any(len(pt & g) / max(1, len(pt | g)) >= 0.5 for g in gtok if g):
            return 'NEAR_MISS'
    E = {norm(e): e for e in ents}
    pe = [E.get(norm(p)) or next((e for k, e in E.items() if k and (k in norm(p))), None) for p in preds]
    if i in ends and any(e is not None and e in ends[i] for e in pe):
        return 'CHAIN_SIBLING'
    if any(e is not None for e in pe):
        return 'OTHER_IN_CTX'
    return 'OUT_OF_CTX'


B = {i: bucket(i, r) for i, r in rows.items()}
n = len(B); order = ['HIT', 'NOT_IN_GRAPH', 'NOT_RETRIEVED', 'NO_ANSWER', 'NEAR_MISS', 'CHAIN_SIBLING', 'OTHER_IN_CTX', 'OUT_OF_CTX']
cnt = collections.Counter(B.values())
inctx = [i for i in B if B[i] not in ('NOT_IN_GRAPH', 'NOT_RETRIEVED')]
print(f'{run_p.split("/")[-1]}  n={n}  Hit {100 * cnt["HIT"] / n:.1f}')
print(f'  gold in context: {len(inctx)} ({100 * len(inctx) / n:.1f}%), Hit given gold in context: {100 * sum(B[i] == "HIT" for i in inctx) / len(inctx):.1f}')
print('  bucket                share of all   share of misses')
miss = n - cnt['HIT']
for b in order:
    print(f'  {b:18s} {cnt[b]:6d} {100 * cnt[b] / n:6.1f}%   {"" if b == "HIT" else f"{100 * cnt[b] / miss:5.1f}%"}')
print('  -- Hit by literal constraint in the gold query (time / order / comparison):')
for flag in (True, False):
    ids = [i for i in B if lit.get(i, False) == flag]; ii = [i for i in ids if i in inctx]
    if ids:
        print(f'     literal={flag!s:5s} n={len(ids):5d}  Hit {100 * sum(B[i] == "HIT" for i in ids) / len(ids):5.1f}   '
              f'Hit|gold in ctx {100 * sum(B[i] == "HIT" for i in ii) / max(1, len(ii)):5.1f}   '
              f'CHAIN_SIBLING {100 * sum(B[i] == "CHAIN_SIBLING" for i in ids) / len(ids):4.1f}%')
print('  -- by question type:')
for t in sorted(set(ctype.get(i, '?') for i in B)):
    ids = [i for i in B if ctype.get(i, '?') == t]; c = collections.Counter(B[i] for i in ids)
    print(f'     {t:12s} n={len(ids):5d}  Hit {100 * c["HIT"] / len(ids):5.1f} | ' + ' '.join(f'{b[:8]} {100 * c[b] / len(ids):4.1f}' for b in order[1:]))
print('  -- by number of gold answers (Hit / F1):')
for lo, hi, nm in [(1, 1, '1'), (2, 3, '2-3'), (4, 10 ** 6, '>=4')]:
    ids = [i for i in B if lo <= len(ref[i].get('a_entity') or []) <= hi]
    if ids:
        print(f'     {nm:4s} n={len(ids):5d}  Hit {100 * sum(rows[i]["hit"] for i in ids) / len(ids):5.1f}  F1 {100 * sum(rows[i]["f1"] for i in ids) / len(ids):5.1f}')
json.dump(B, open(run_p.replace('.jsonl', '.anatomy.json'), 'w'))
