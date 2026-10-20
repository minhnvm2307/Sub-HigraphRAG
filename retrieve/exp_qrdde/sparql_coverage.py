"""Coverage of the gold-query labels (sparql_labels.py) on our 2-hop subgraphs, before any retraining (CPU, no LLM).

Labels:
  WebQSP  read straight from the annotation (Parses[0]: TopicEntityName, InferentialChain, Constraints, Order): its time
          constraints live inside FILTER(NOT EXISTS .. EXISTS {..}), which parse_sparql puts in `filters`, not patterns.
  CWQ     parse_sparql + extract_label on the gold SPARQL (heuristic anchors). SPARQL uses Freebase MIDs and our records
          use names for text entities, so MIDs are mapped to names from the WebQSP annotations (topics, answers, entity
          constraints) and, for a topic still unmapped, the record's single unused topic name. A constraint whose object
          MID has no name is matched on its predicate only (counted).
Reported per dataset/split, on records of exp_qrdde/data/<ds>/<split>.pkl:
  chain_complete          the gold chain can be walked from the topic inside the subgraph (ceiling of the new labels)
  answers                 n_answer_candidates (chain end nodes) vs gold answers in the graph; gold recall of candidates
  constraint triples      per question; share currently NEGATIVE under the SP labels (label == 0)
  hops                    topological vs semantic (CVT steps not counted) chain length
  reach@T=3               share of chain / constraint triples an RBF walk of T=3 steps from a topic can score
                          (an endpoint within 2 undirected hops of a topic)
  python sparql_coverage.py webqsp train   |   python sparql_coverage.py cwq test
"""
import collections, json, pickle, sys
import numpy as np
from sparql_labels import parse_sparql, extract_label, bind_to_subgraph, MID

ROOT = '/home/minhnv/Lab/SubgraphRAG'
WQ = f'{ROOT}/relrag/data/webqsp_orig/WebQSP/data'


def webqsp_annotations():
    out, mid2name = {}, {}
    for split in ('train', 'test'):
        for q in json.load(open(f'{WQ}/WebQSP.{split}.json'))['Questions']:
            for p in q.get('Parses', []):
                if p.get('TopicEntityMid'):
                    mid2name[p['TopicEntityMid']] = p['TopicEntityName']
                for a in p.get('Answers', []):
                    if a.get('AnswerType') == 'Entity' and a.get('EntityName'):
                        mid2name[a['AnswerArgument']] = a['EntityName']
                for c in p.get('Constraints', []):
                    if c.get('ArgumentType') == 'Entity' and c.get('EntityName'):
                        mid2name[c['Argument']] = c['EntityName']
            p = next((p for p in q.get('Parses', []) if p.get('InferentialChain')), None)
            if p is None:
                continue
            cons = [{'predicate': c['NodePredicate'],
                     'object': c['EntityName'] if c.get('ArgumentType') == 'Entity' and c.get('EntityName') else '?v',
                     'kind': 'value' if c.get('ArgumentType') == 'Entity' else 'literal'} for c in p.get('Constraints', [])]
            if p.get('Order') and p['Order'].get('NodePredicate'):
                cons.append({'predicate': p['Order']['NodePredicate'], 'object': '?v', 'kind': 'order_key'})
            out[q['QuestionId']] = {'ok': True, 'topic': p['TopicEntityName'], 'constraints': cons,
                                    'chain': [{'predicate': r, 'direction': 'fwd'} for r in p['InferentialChain']]}
    return out, mid2name


def cwq_labels(split, recs, mid2name):
    out, why = {}, collections.Counter()
    raw = [q for s in ('train', 'dev', 'test')             # our 'val' = CWQ dev; ids are unique across files
           for q in json.load(open(f'{ROOT}/relrag/data/cwq_orig/ComplexWebQuestions_{s}.json'))]
    for q in raw:
        rec = recs.get(q['ID'])
        if rec is None:
            continue
        lab = extract_label(parse_sparql(q['sparql']))
        if not lab.get('ok'):
            why[lab['reason']] += 1; continue
        topics = [rec['entity_names'][i] for i in rec['q_ent']]
        mids = [lab['topic']] + [c['anchor'] for c in lab['other_anchor_chains']]
        known = {m: mid2name[m] for m in mids if m in mid2name}
        left = [t for t in topics if t not in known.values()]
        if lab['topic'] not in known and len(left) == 1:
            known[lab['topic']] = left[0]
        if lab['topic'] not in known:
            why['topic MID without a name'] += 1; continue
        cons = []
        for c in lab['constraints']:
            o = c['object'] if not c['attaches_to'] or c['object'] not in c['attaches_to'] else c['subject']
            if MID.match(o):
                o = known.get(o, mid2name.get(o, '?unmapped'))
            elif o.startswith('?'):
                o = '?v'
            cons.append({'predicate': c['predicate'], 'object': o, 'kind': c['kind']})
        out[q['ID']] = {'ok': True, 'topic': known[lab['topic']], 'constraints': cons, 'chain': lab['chain']}
    return out, why


def keep_constraints(lab, cons_triples):
    """bind_to_subgraph accepts any object unless the object is a MID; our objects are names (or '?v'), so keep a
    constraint triple only if a constraint with its predicate has a variable object or one matching the triple"""
    keep = set()
    for tr in cons_triples:
        for c in lab['constraints']:
            if c['predicate'] == tr[1] and (c['object'].startswith('?') or c['object'] in (tr[0], tr[2])):
                keep.add(tr); break
    return keep


def dist_from(rec, sources):
    adj = collections.defaultdict(list)
    for h, t in zip(rec['h'].tolist(), rec['t'].tolist()):
        adj[h].append(t); adj[t].append(h)
    d = {s: 0 for s in sources}; dq = collections.deque(sources)
    while dq:
        u = dq.popleft()
        if d[u] >= 3:
            continue
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; dq.append(v)
    return d


def main():
    ds, split = sys.argv[1], sys.argv[2]
    recs = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/exp_qrdde/data/{ds}/{split}.pkl', 'rb'))}
    wq, mid2name = webqsp_annotations()
    if ds == 'webqsp':
        labels, why = {i: l for i, l in wq.items() if i in recs}, collections.Counter()
    else:
        labels, why = cwq_labels(split, recs, mid2name)
    S = collections.defaultdict(list)
    n_unmapped_obj = 0
    for qid, lab in labels.items():
        rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
        trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
        idx = {tr: k for k, tr in enumerate(trip)}
        n_unmapped_obj += sum(c['object'] == '?unmapped' for c in lab['constraints'])
        lab = dict(lab, constraints=[dict(c, object='?v' if c['object'] == '?unmapped' else c['object']) for c in lab['constraints']])
        b = bind_to_subgraph(lab, trip, named_ids=set(N[:rec['n_text']]))
        S['complete'].append(b['chain_complete'])
        if not b['chain_complete']:
            continue
        gold = {N[i] for i in rec.get('a_ent', [])}
        S['n_cand'].append(b['n_answer_candidates']); S['n_gold'].append(len(gold))
        S['cand_gt_gold'].append(b['n_answer_candidates'] > len(gold))
        S['topo'].append(b['n_topological_hops']); S['sem'].append(b['n_semantic_hops'])
        lab_sp = rec['label'].astype(bool)
        ch = [idx[t] for t in b['chain_triples'] if t in idx]; co = [idx[t] for t in keep_constraints(lab, b['constraint_triples']) if t in idx]
        S['n_chain_trip'].append(len(ch)); S['n_cons_trip'].append(len(co)); S['has_cons'].append(len(co) > 0)
        S['chain_in_sp'] += [bool(lab_sp[k]) for k in ch]; S['cons_neg_sp'] += [not lab_sp[k] for k in co]
        S['sp_in_chain'].append(len(set(np.nonzero(lab_sp)[0].tolist()) & set(ch)) / max(lab_sp.sum(), 1))
        d = dist_from(rec, rec['q_ent'])
        ok = lambda k: min(d.get(int(rec['h'][k]), 9), d.get(int(rec['t'][k]), 9)) <= 2
        S['chain_reach'] += [ok(k) for k in ch]; S['cons_reach'] += [ok(k) for k in co]
        bound_preds = {trip[k][1] for k in co}
        S['cons_kind_bound'] += [(c['kind'], c['predicate'] in bound_preds) for c in lab['constraints']]
    m = lambda k: 100 * np.mean(S[k]) if S[k] else float('nan')
    n = len(S['complete'])
    print(f'== {ds} {split}: {len(recs)} records, {n} with a gold label ({dict(why)})')
    print(f'chain_complete: {m("complete"):.1f}% of labelled questions')
    print(f'answer candidates per question: mean {np.mean(S["n_cand"]):.1f}, median {np.median(S["n_cand"]):.0f} vs gold answers '
          f'in graph mean {np.mean(S["n_gold"]):.1f}; candidates > gold in {m("cand_gt_gold"):.1f}% of questions')
    print(f'chain triples per question: mean {np.mean(S["n_chain_trip"]):.1f}; {m("chain_in_sp"):.1f}% of them are SP-positive; '
          f'chain covers {m("sp_in_chain"):.1f}% of the SP-positive triples')
    print(f'constraint triples per question: mean {np.mean(S["n_cons_trip"]):.1f}, questions with >=1: {m("has_cons"):.1f}%; '
          f'{m("cons_neg_sp"):.1f}% of them are NEGATIVE under SP labels; constraint objects without a name: {n_unmapped_obj}')
    c = collections.Counter(zip(S['topo'], S['sem']))
    print('hops (topological, semantic): ' + ', '.join(f'{k}: {v}' for k, v in sorted(c.items(), key=lambda x: -x[1])[:8]))
    print(f'reach@T=3: chain triples {m("chain_reach"):.1f}%, constraint triples {m("cons_reach"):.1f}%')
    kr = collections.defaultdict(list)
    for k, v in S['cons_kind_bound']:
        kr[k].append(v)
    print('gold constraints found in the subgraph, by kind: ' + ', '.join(f'{k} {100*np.mean(v):.1f}% of {len(v)}' for k, v in kr.items()))


if __name__ == '__main__':
    main()
