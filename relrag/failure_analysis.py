"""Why do retrievers miss gold evidence? Per-triple and per-question failure analysis on exported rankings.

For every GPT-4o evidence triple that exists in the question's subgraph (text match), record its rank under each
retriever and its structural category:
  dist    min BFS hop distance (undirected) of its endpoints to the topic entities (0 = touches a topic)
  orient  'out' if the endpoint nearer to the topic is the head, 'in' otherwise (triple points toward the topic)
  cvt     an endpoint is a non-text node (CVT / mediator, all share ONE learned embedding in SubgraphRAG's MLP)
  fanout  # triples sharing the near endpoint and the relation (siblings competing for the same slot)
Then: recall@k of gold triples per category and retriever; the hop-distance profile of what fills each retriever's
top-k vs the gold profile; and, for questions whose answer is missed at k, the answer distance, answer count, CWQ
composition type, and the best rank of any answer-touching triple.

  python failure_analysis.py --dataset cwq  > runs/analysis/failure_cwq.md
  python failure_analysis.py --dataset webqsp > runs/analysis/failure_webqsp.md
"""
import argparse, json, pickle
from collections import defaultdict, deque, Counter
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
QD = f'{ROOT}/retrieve/exp_qrdde/runs'
RR = f'{ROOT}//relrag/runs/path_codefaithful'
CFG = {
    'cwq': dict(raw=f'{ROOT}/retrieve/data_files/cwq/processed/test.pkl',
                gpt=f'{ROOT}/reason/gpt_labeled/gpt_labeled_cwq_cleaned.pt',
                ref=f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth',
                orig=f'{ROOT}//relrag/data/cwq_orig/ComplexWebQuestions_test.json',
                runs={'AER (WebQSP->CWQ)': f'{QD}/webqsp_dde_aer/retrieval_cwq_test.pth',
                      'SubgraphRAG repro (WebQSP->CWQ)': f'{QD}/webqsp_dde/retrieval_cwq_test.pth',
                      'REL-RAG P (WebQSP->CWQ)': f'{RR}/retrieval_cwq_test_path_bf/retrieval_result_500.pth',
                      'SubgraphRAG official (CWQ-trained)': 'REF'}),
    'webqsp': dict(raw=f'{ROOT}/retrieve/data_files/webqsp/processed/test.pkl',
                   gpt=f'{ROOT}/retrieve/data_files/webqsp/gpt_triples.pth',
                   ref=f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth', orig=None,
                   runs={'AER': f'{QD}/webqsp_dde_aer/retrieval_test.pth',
                         'SubgraphRAG official': 'REF',
                         'REL-RAG P': f'{RR}/retrieval_test_path_bf/retrieval_result_500.pth'}),
}
INF = 99


def bfs(n, edges, src):
    adj = defaultdict(list)
    for h, t in edges:
        adj[h].append(t); adj[t].append(h)
    d = np.full(n, INF); q = deque()
    for s in src:
        d[s] = 0; q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] == INF:
                d[v] = d[u] + 1; q.append(v)
    return d


def bucket_fan(f):
    return '1' if f == 1 else '2-10' if f <= 10 else '11-100' if f <= 100 else '>100'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cwq', choices=list(CFG))
    ap.add_argument('--ks', default='50,100')
    ap.add_argument('--examples', type=int, default=12)
    ap.add_argument('--first', default=None, help='name=path: put this export first (replaces the first run)')
    args = ap.parse_args()
    cfg = CFG[args.dataset]; ks = [int(k) for k in args.ks.split(',')]
    if args.first:
        nm, p = args.first.split('=', 1); cfg['runs'] = {nm: p, **{k: v for k, v in list(cfg['runs'].items())[1:]}}
    raw = {s['id']: s for s in pickle.load(open(cfg['raw'], 'rb'))}
    ref = torch.load(cfg['ref'], weights_only=False)
    runs = {n: (ref if p == 'REF' else torch.load(p, weights_only=False)) for n, p in cfg['runs'].items()}
    gpt = torch.load(cfg['gpt'], weights_only=False)
    ctype = {}
    if cfg['orig']:
        ctype = {x['ID']: x['compositionality_type'] for x in json.load(open(cfg['orig']))}
    ids = [i for i in raw if i in ref and ref[i]['a_entity_in_graph'] and all(i in r for r in runs.values())]
    names_of = lambda s: list(s['text_entity_list']) + list(s['non_text_entity_list'])

    rows = []          # one per gold triple
    fill = {n: {k: Counter() for k in ks} for n in runs}     # dist profile of top-k
    gold_prof = Counter()
    qrows = []         # one per question
    for i in ids:
        s = raw[i]; N = names_of(s); ntext = len(s['text_entity_list']); R = s['relation_list']
        H, Rr, T = s['h_id_list'], s['r_id_list'], s['t_id_list']
        if not H or not s['q_entity_id_list']:
            continue
        d = bfs(len(N), list(zip(H, T)), s['q_entity_id_list'])
        trip = [(N[h], R[r], N[t]) for h, r, t in zip(H, Rr, T)]
        info = {}
        sib = Counter()
        for h, r, t in zip(H, Rr, T):
            near = h if d[h] <= d[t] else t
            sib[(near, r)] += 1
        for (h, r, t), x in zip(zip(H, Rr, T), trip):
            near = h if d[h] <= d[t] else t
            info.setdefault(x, dict(dist=int(min(d[h], d[t])), orient='out' if d[h] <= d[t] else 'in',
                                    cvt=bool(h >= ntext or t >= ntext), fan=sib[(near, r)]))
        rank = {}
        for n, r in runs.items():
            rk = {}
            for j, y in enumerate(r[i]['scored_triples']):
                rk.setdefault(tuple(y[:3]), j)
            rank[n] = rk
            for k in ks:
                for y in r[i]['scored_triples'][:k]:
                    fill[n][k][min(info.get(tuple(y[:3]), {'dist': INF})['dist'], 3)] += 1
        G = {tuple(t[:3]) for t in gpt.get(i, [])} & set(info)
        for x in G:
            gold_prof[min(info[x]['dist'], 3)] += 1
            rows.append(dict(id=i, triple=x, **info[x], ctype=ctype.get(i, '-'),
                             **{f'rank|{n}': rank[n].get(x, 10 ** 6) for n in runs}))
        # answers
        A = set(ref[i]['a_entity_in_graph']); aid = [j for j, nm in enumerate(N) if nm in A]
        ad = min([d[j] for j in aid], default=INF)
        q = dict(id=i, q=s['question'], nans=len(A), adist=int(ad), ctype=ctype.get(i, '-'), ngold=len(G))
        for n, r in runs.items():
            best = min([j for j, y in enumerate(r[i]['scored_triples']) if y[0] in A or y[2] in A], default=10 ** 6)
            q[f'ansrank|{n}'] = best
        qrows.append(q)

    names = list(runs)
    print(f'## Failure analysis — {args.dataset.upper()} test ({len(qrows)} questions, {len(rows)} gold GPT triples present in the subgraph)\n')
    print('Rank = position of the gold triple in the retriever\'s list (REL-RAG and transfer exports stop at 500, ' +
          ('WebQSP AER at 200' if args.dataset == 'webqsp' else 'SubgraphRAG official is the full list') + ').\n')

    def table(title, key, order=None):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        keys = order or sorted(groups, key=lambda g: -len(groups[g]))
        print(f'### Gold-triple recall by {title}\n')
        print('| ' + title + ' | n (%) | ' + ' | '.join(f'{n} @{k}' for n in names for k in ks) + ' |')
        print('|---|---|' + '---|' * (len(names) * len(ks)))
        for g in keys:
            if g not in groups:
                continue
            v = groups[g]
            cells = [f'{np.mean([r[f"rank|{n}"] < k for r in v]) * 100:.1f}' for n in names for k in ks]
            print(f'| {g} | {len(v)} ({len(v) / len(rows) * 100:.0f}%) | ' + ' | '.join(cells) + ' |')
        print()

    for r in rows:
        r['dist_b'] = {0: '0 (touches topic)', 1: '1', 2: '2'}.get(r['dist'], '3+')
        r['fan_b'] = bucket_fan(r['fan'])
        r['cvt_b'] = 'CVT endpoint' if r['cvt'] else 'text-only'
        r['combo'] = f"d{min(r['dist'], 3)} {r['orient']} {'cvt' if r['cvt'] else 'txt'}"
    table('hop distance', 'dist_b', ['0 (touches topic)', '1', '2', '3+'])
    table('orientation', 'orient', ['out', 'in'])
    table('CVT', 'cvt_b')
    table('sibling fan-out', 'fan_b', ['1', '2-10', '11-100', '>100'])
    table('distance x orientation x CVT', 'combo')
    if ctype:
        table('CWQ type', 'ctype')

    print('### What fills the top-k: hop-distance profile of retrieved triples vs gold (%)\n')
    print('| | k | dist 0 | dist 1 | dist 2 | dist 3+ |')
    print('|---|---|---|---|---|---|')
    tot = sum(gold_prof.values())
    print('| gold GPT evidence | – | ' + ' | '.join(f'{gold_prof[j] / tot * 100:.1f}' for j in range(4)) + ' |')
    for n in names:
        for k in ks:
            c = fill[n][k]; t = sum(c.values())
            print(f'| {n} | {k} | ' + ' | '.join(f'{c[j] / t * 100:.1f}' for j in range(4)) + ' |')
    print()

    print('### Questions whose answer is missed at k=50 (no answer-touching triple in the top 50)\n')
    print('| retriever | missed | answer dist 1 / 2 / 3+ / unreachable (%) | median #answers (missed vs hit) | best answer-triple rank among missed: ≤100 / ≤500 / >500 (%) |')
    print('|---|---|---|---|---|')
    for n in names:
        miss = [q for q in qrows if q[f'ansrank|{n}'] >= 50]; hit = [q for q in qrows if q[f'ansrank|{n}'] < 50]
        ad = Counter(min(q['adist'], 4) if q['adist'] < INF else 9 for q in miss); m = max(len(miss), 1)
        rk = [q[f'ansrank|{n}'] for q in miss]
        print(f'| {n} | {len(miss)} ({len(miss) / len(qrows) * 100:.1f}%) | '
              f'{ad[1] / m * 100:.0f} / {ad[2] / m * 100:.0f} / {(ad[3] + ad[4]) / m * 100:.0f} / {ad[9] / m * 100:.0f} | '
              f'{np.median([q["nans"] for q in miss]) if miss else 0:.0f} vs {np.median([q["nans"] for q in hit]):.0f} | '
              f'{np.mean([x < 100 for x in rk]) * 100:.0f} / {np.mean([x < 500 for x in rk]) * 100:.0f} / {np.mean([x >= 500 for x in rk]) * 100:.0f} |')
    print()
    if ctype:
        print('### Answer missed at k=50 by CWQ type (% of questions of that type)\n')
        print('| type | n | ' + ' | '.join(names) + ' |')
        print('|---|---|' + '---|' * len(names))
        for t in ['composition', 'conjunction', 'comparative', 'superlative']:
            v = [q for q in qrows if q['ctype'] == t]
            print(f'| {t} | {len(v)} | ' + ' | '.join(f'{np.mean([q[f"ansrank|{n}"] >= 50 for q in v]) * 100:.1f}' for n in names) + ' |')
        print()

    # examples: gold triples the first retriever misses at 100 but the last (reference) gets in its top 50
    a, b = names[0], names[-1]
    ex = [r for r in rows if r[f'rank|{a}'] >= 100 and r[f'rank|{b}'] < 50]
    print(f'### Examples: gold triples missed by {a} (rank ≥ 100) but in the top 50 of {b} ({len(ex)} such triples)\n')
    rng = np.random.default_rng(0)
    qtext = {q['id']: q['q'] for q in qrows}
    for r in [ex[j] for j in rng.choice(len(ex), min(args.examples, len(ex)), replace=False)] if ex else []:
        rk = r[f'rank|{a}']
        print(f'- *{qtext[r["id"]]}* [{r["ctype"]}] `{r["triple"]}` — dist {r["dist"]}, {r["orient"]}, '
              f'{"CVT" if r["cvt"] else "text"}, fan-out {r["fan"]}; rank {a}: {">500" if rk >= 10 ** 6 else rk}, {b}: {r[f"rank|{b}"]}')
    print()
    json.dump({'rows': [{**r, 'triple': list(r['triple'])} for r in rows], 'questions': qrows},
              open(f'{ROOT}//relrag/runs/analysis/failure_{args.dataset}.json', 'w'))


if __name__ == '__main__':
    main()
