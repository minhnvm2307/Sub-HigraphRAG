"""Gold-query labels v2 (10-17). v1 (label_compare.py -> sparql_<split>.pkl) marked EVERY trail that follows the gold
relation chain from the topic as positive, including trails ending at entities that are not answers: in the real
query a FILTER (mostly time, absent from our subgraphs) removes them, so v1 taught the retriever to fetch complete
chains to wrong candidates, which mislead the LLM (docs/12 sections 18-20). v2 keeps:
  chain  triples of the trails that END AT A GOLD ANSWER
  cons   constraint triples attached to a node of those trails (object matched by name, '?v' = any)
  neg    triples that appear only on trails ending at a non-answer (hard negatives; saved, unused by default)
Questions whose chain reaches no gold answer keep SP (handled in qrdde.apply_sparql_labels).
Output data/<ds>/sparqlv2_<split>.pkl = {id: {'chain': [idx], 'cons': [idx], 'neg': [idx]}} + statistics vs v1.
  python make_sparql_v2.py <webqsp|cwq> <train|val|test>
"""
import collections, pickle, sys
import numpy as np
import sparql_coverage as C

D = '/home/minhnv/Lab/SubgraphRAG/retrieve/exp_qrdde/data'


def walk(lab, trip):
    by_sp, by_po = collections.defaultdict(list), collections.defaultdict(list)
    for h, r, t in trip:
        by_sp[(h, r)].append(t); by_po[(r, t)].append(h)
    front = {lab['topic']: []}
    for hop in lab['chain']:
        nxt = {}
        for node, tr in front.items():
            if hop['direction'] == 'fwd':
                for t in by_sp.get((node, hop['predicate']), []):
                    nxt.setdefault(t, tr + [(node, hop['predicate'], t)])
            else:
                for h in by_po.get((hop['predicate'], node), []):
                    nxt.setdefault(h, tr + [(h, hop['predicate'], node)])
        front = nxt
        if not front:
            break
    return front, by_sp, by_po


def all_trails(lab, trip):
    """every trail (not one per end node): an end node reached through several CVTs keeps all of them"""
    by_sp, by_po = collections.defaultdict(list), collections.defaultdict(list)
    for h, r, t in trip:
        by_sp[(h, r)].append(t); by_po[(r, t)].append(h)
    trails = [(lab['topic'], [])]
    for hop in lab['chain']:
        nxt = []
        for node, tr in trails:
            if hop['direction'] == 'fwd':
                nxt += [(t, tr + [(node, hop['predicate'], t)]) for t in by_sp.get((node, hop['predicate']), [])]
            else:
                nxt += [(h, tr + [(h, hop['predicate'], node)]) for h in by_po.get((hop['predicate'], node), [])]
        trails = nxt[:5000]
        if not trails:
            break
    return trails, by_sp, by_po


def constraints_on(lab, nodes, by_sp, by_po):
    out = set()
    for c in lab['constraints']:
        o = '?v' if c['object'] == '?unmapped' else c['object']
        for n in nodes:
            for t in by_sp.get((n, c['predicate']), []):
                if o.startswith('?') or o == t:
                    out.add((n, c['predicate'], t))
            for h in by_po.get((c['predicate'], n), []):
                if o.startswith('?') or o == h:
                    out.add((h, c['predicate'], n))
    return out


def main():
    ds, split = sys.argv[1], sys.argv[2]
    recs = {r['id']: r for r in pickle.load(open(f'{D}/{ds}/{split}.pkl', 'rb'))}
    wq, m2n = C.webqsp_annotations()
    labels = {i: l for i, l in wq.items() if i in recs} if ds == 'webqsp' else C.cwq_labels(split, recs, m2n)[0]
    v1 = pickle.load(open(f'{D}/{ds}/sparql_{split}.pkl', 'rb'))
    out, S = {}, collections.defaultdict(list)
    for qid, lab in labels.items():
        rec = recs[qid]; N, L = rec['entity_names'], rec['relation_list']
        gold = {N[a] for a in rec.get('a_ent', [])}
        trip = [(N[h], L[r], N[t]) for h, r, t in zip(rec['h'].tolist(), rec['r'].tolist(), rec['t'].tolist())]
        ix = {}
        for k, tr in enumerate(trip):
            ix.setdefault(tr, k)
        trails, by_sp, by_po = all_trails(lab, trip)
        good = [tr for end, tr in trails if end in gold]
        bad = [tr for end, tr in trails if end not in gold and not C.MID.match(end) and not end.startswith(('m.', 'g.'))]
        if not good:
            continue
        chain = {t for tr in good for t in tr}
        nodes = {lab['topic']} | {x for t in chain for x in (t[0], t[2])}
        cons = constraints_on(lab, nodes, by_sp, by_po) - chain
        neg = {t for tr in bad for t in tr} - chain - cons
        out[qid] = {'chain': sorted(ix[t] for t in chain if t in ix), 'cons': sorted(ix[t] for t in cons if t in ix),
                    'neg': sorted(ix[t] for t in neg if t in ix)}
        o1 = v1.get(qid, {'chain': [], 'cons': []})
        S['v1'].append(len(o1['chain']) + len(o1['cons'])); S['v2'].append(len(out[qid]['chain']) + len(out[qid]['cons']))
        S['neg'].append(len(out[qid]['neg'])); S['had_bad'].append(bool(bad))
        S['removed'].append(len(set(o1['chain']) - set(out[qid]['chain'])))
    pickle.dump(out, open(f'{D}/{ds}/sparqlv2_{split}.pkl', 'wb'))
    n = len(recs)
    print(f'{ds} {split}: v2 label for {len(out)}/{n} records (v1: {sum(1 for v in v1.values() if v["chain"])}); '
          f'labelled triples per question v1 {np.mean(S["v1"]):.1f} -> v2 {np.mean(S["v2"]):.1f}; '
          f'questions whose chain also reaches non-answers: {100 * np.mean(S["had_bad"]):.0f}%; '
          f'v1 chain triples dropped per question {np.mean(S["removed"]):.1f}; hard negatives per question {np.mean(S["neg"]):.1f}')


if __name__ == '__main__':
    main()
