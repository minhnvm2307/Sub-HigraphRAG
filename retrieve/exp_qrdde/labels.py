"""Training labels from the GOLD QUERY (SPARQL) of WebQSP / CWQ, bound to each question's subgraph.

The default SubgraphRAG label is "every triple on a shortest path topic -> answer" (SP), which is noisy: a shortest path
is often not the reasoning of the question. The gold query says what the question really asks:
  chain        the predicate chain from the topic entity to the answer variable (WebQSP: InferentialChain)
  constraints  the other patterns of the query (type / entity value / order key), attached to nodes of the chain

Three label versions, each written to exp_qrdde/data/<dataset>/<file>_<split>.pkl as {question id: {...}} with triple
indices into that record's (h, r, t) arrays:
  v1  sparql_<split>.pkl    every trail that follows the gold chain from the topic (one trail per end node), plus the
                            constraint triples on its nodes. Trails ending at NON-answers are positives too (in the
                            real query a FILTER, absent from our subgraphs, removes them).
  v2  sparqlv2_<split>.pkl  only the trails that END AT A GOLD ANSWER (+ their constraint triples); triples found only
                            on trails to non-answers are saved as hard negatives ('neg', unused by default).
  v3  sparqlv3_<split>.pkl  v2, plus for questions v2 cannot label: a hop walked in the reverse direction when the
                            declared one has no triple ('flip'), or the gold trails to the deepest node reached plus the
                            shortest bridges (<= 3 hops) from there to a gold answer ('bridge').
Questions a version cannot label keep the SP label at training time (qrdde.apply_sparql_labels).

The same module gives eval.py its gold-query targets on the test split (gold_query_targets).

Usage (CPU only, no LLM):
  python labels.py --dataset webqsp --split train --version v2
  (v2 needs nothing else; v3 reads the v2 file of the same split, so build v2 first)

Sources:
  WebQSP  relrag/data/webqsp_orig/WebQSP/data/WebQSP.{train,test}.json, Parses[0]: TopicEntityName, InferentialChain,
          Constraints, Order (read directly, no SPARQL parsing needed)
  CWQ     relrag/data/cwq_orig/ComplexWebQuestions_{train,dev,test}.json, field 'sparql' (parsed here). CWQ queries use
          Freebase MIDs while our records use entity names, so MIDs are mapped to names via the WebQSP annotations;
          a topic MID still unmapped takes the record's single topic name not used by another anchor.
"""
import argparse, collections, json, pickle, re

ROOT = '/home/minhnv/Lab/SubgraphRAG'
DATA = f'{ROOT}/retrieve/exp_qrdde/data'
WEBQSP_JSON = f'{ROOT}/relrag/data/webqsp_orig/WebQSP/data'
CWQ_JSON = f'{ROOT}/relrag/data/cwq_orig'
LABEL_FILE = {'v1': 'sparql', 'v2': 'sparqlv2', 'v3': 'sparqlv3'}      # data/<ds>/<file>_<split>.pkl

FREEBASE_MID = re.compile(r'^m\.[0-9a-z_]+$')
TYPE_PREDICATES = ('common.topic.notable_types', 'type.object.type')
MAX_TRAILS = 5000          # cap on trails kept per hop (a few hub entities explode otherwise)
MAX_BRIDGE_HOPS = 3        # v3: longest bridge from the chain frontier to a gold answer


# ============================================================ 1. read the gold queries
def _split_where_clauses(sparql):
    """Split the WHERE body into triple-pattern clauses; FILTER / OPTIONAL / UNION / MINUS blocks go to `filters`.
    A clause ends at a '.' surrounded by whitespace (Freebase predicates contain dots themselves)."""
    start = sparql.find('WHERE')
    if start < 0:
        return [], ['<no WHERE>']
    body, depth, kept = sparql[sparql.find('{', start) + 1:], 0, []
    for ch in body:                                            # cut at the brace closing WHERE {
        if ch == '{':
            depth += 1
        elif ch == '}':
            if depth == 0:
                break
            depth -= 1
        kept.append(ch)
    body = ''.join(kept)
    keywords = ('FILTER', 'OPTIONAL', 'UNION', 'MINUS')
    clauses, filters, current = [], [], []
    i, n = 0, len(body)
    while i < n:
        keyword = next((k for k in keywords if body[i:i + 8].upper().startswith(k)), None)
        if keyword and (i == 0 or not body[i - 1].isalnum()):
            j = i + len(keyword)
            while j < n and body[j] not in '({':
                j += 1
            if j < n:                                          # skip the whole bracketed block
                opener, closer = ('(', ')') if body[j] == '(' else ('{', '}')
                level, k = 0, j
                while k < n:
                    if body[k] == opener:
                        level += 1
                    elif body[k] == closer:
                        level -= 1
                        if level == 0:
                            break
                    k += 1
                filters.append(' '.join(body[i:k + 1].split()))
                i = k + 1
                continue
        if body[i] == '.' and (i == 0 or body[i - 1].isspace()) and (i + 1 >= n or body[i + 1].isspace()):
            clauses.append(''.join(current).strip()); current = []
        else:
            current.append(body[i])
        i += 1
    if ''.join(current).strip():
        clauses.append(''.join(current).strip())
    return [c for c in clauses if c], filters


def parse_sparql(sparql):
    """-> {select_var, patterns: [(s, p, o)], order_by: {direction, var} | None} ('ns:' prefixes removed)"""
    select = re.search(r'SELECT\s+(?:DISTINCT\s+)?(\?\w+)', sparql)
    order = re.search(r'ORDER\s+BY\s+(DESC|ASC)?\s*\(?\s*(?:xsd:\w+\s*\()?\s*(\?\w+)', sparql, re.I)
    clauses, _ = _split_where_clauses(sparql)
    patterns = []
    for clause in clauses:
        tokens = clause.split()
        if len(tokens) == 3:
            patterns.append(tuple(t[3:] if t.startswith('ns:') else t for t in tokens))
    return {'select_var': select.group(1) if select else None, 'patterns': patterns,
            'order_by': {'direction': (order.group(1) or 'ASC').upper(), 'var': order.group(2)} if order else None}


def chain_and_constraints_from_sparql(parsed):
    """Split the patterns of a parsed CWQ query into the chain (anchor entity -> answer variable) and constraints.
    Anchors = MIDs that are not type objects; when several anchors reach the answer, the LONGEST chain is the main one
    (in CWQ compositions the named entity sits far from ?x, type / value constraints sit next to it).
    -> {'ok': True, topic (MID), chain: [{predicate, direction}], other_anchors: [MID], constraints: [...]}
       or {'ok': False, 'reason': ...}"""
    patterns, answer_var = parsed['patterns'], parsed['select_var']
    if not patterns or answer_var is None:
        return {'ok': False, 'reason': 'no pattern or no SELECT variable'}
    type_objects = {o for s, p, o in patterns if p in TYPE_PREDICATES}
    anchors = sorted({x for s, p, o in patterns for x in (s, o) if FREEBASE_MID.match(x) and x not in type_objects})
    if not anchors:
        return {'ok': False, 'reason': 'no anchor entity in WHERE'}
    neighbours = collections.defaultdict(list)                 # node -> [(other node, pattern index)]
    for i, (s, p, o) in enumerate(patterns):
        neighbours[s].append((o, i)); neighbours[o].append((s, i))

    def pattern_path_to_answer(anchor):                        # BFS over patterns: anchor -> answer variable
        previous, queue = {anchor: None}, collections.deque([anchor])
        while queue:
            node = queue.popleft()
            if node == answer_var:
                break
            for other, i in neighbours[node]:
                if other not in previous:
                    previous[other] = (node, i); queue.append(other)
        if answer_var not in previous:
            return None
        path, node = [], answer_var
        while previous[node] is not None:
            node, i = previous[node]; path.append(i)
        return path[::-1]

    found = [(a, path) for a in anchors if (path := pattern_path_to_answer(a)) is not None]
    if not found:
        return {'ok': False, 'reason': 'no path from an anchor to the answer variable'}
    found.sort(key=lambda anchor_path: -len(anchor_path[1]))
    topic, chain_idx = found[0]
    chain, chain_nodes, node = [], [topic], topic
    for i in chain_idx:
        s, p, o = patterns[i]
        forward = s == node
        chain.append({'predicate': p, 'direction': 'fwd' if forward else 'rev'})
        node = o if forward else s; chain_nodes.append(node)
    order_var = parsed['order_by']['var'] if parsed['order_by'] else None
    constraints = []
    for i, (s, p, o) in enumerate(patterns):
        if i in chain_idx:
            continue
        attaches_to = [x for x in (s, o) if x in chain_nodes]
        kind = ('type' if p.endswith('notable_types') or p == 'type.object.type' else
                'order_key' if order_var in (s, o) else 'value' if FREEBASE_MID.match(o) else 'other')
        constraints.append({'predicate': p, 'subject': s, 'object': o, 'attaches_to': attaches_to, 'kind': kind})
    return {'ok': True, 'topic': topic, 'chain': chain, 'other_anchors': [a for a, _ in found[1:]],
            'constraints': constraints}


def webqsp_gold_queries():
    """-> ({question id: gold query}, {MID: entity name}) from the WebQSP annotations (train + test).
    gold query = {'topic': name, 'chain': [{predicate, direction}], 'constraints': [{predicate, object, kind}]}
    with object = an entity name, or '?v' for a literal / an order key (any object accepted)."""
    queries, mid_to_name = {}, {}
    for split in ('train', 'test'):
        for q in json.load(open(f'{WEBQSP_JSON}/WebQSP.{split}.json'))['Questions']:
            for parse in q.get('Parses', []):
                if parse.get('TopicEntityMid'):
                    mid_to_name[parse['TopicEntityMid']] = parse['TopicEntityName']
                for a in parse.get('Answers', []):
                    if a.get('AnswerType') == 'Entity' and a.get('EntityName'):
                        mid_to_name[a['AnswerArgument']] = a['EntityName']
                for c in parse.get('Constraints', []):
                    if c.get('ArgumentType') == 'Entity' and c.get('EntityName'):
                        mid_to_name[c['Argument']] = c['EntityName']
            parse = next((p for p in q.get('Parses', []) if p.get('InferentialChain')), None)
            if parse is None:
                continue
            constraints = [{'predicate': c['NodePredicate'],
                            'object': c['EntityName'] if c.get('ArgumentType') == 'Entity' and c.get('EntityName') else '?v',
                            'kind': 'value' if c.get('ArgumentType') == 'Entity' else 'literal'}
                           for c in parse.get('Constraints', [])]
            if parse.get('Order') and parse['Order'].get('NodePredicate'):
                constraints.append({'predicate': parse['Order']['NodePredicate'], 'object': '?v', 'kind': 'order_key'})
            queries[q['QuestionId']] = {'topic': parse['TopicEntityName'], 'constraints': constraints,
                                        'chain': [{'predicate': r, 'direction': 'fwd'} for r in parse['InferentialChain']]}
    return queries, mid_to_name


def cwq_gold_queries(records, mid_to_name):
    """-> {question id: gold query} for the CWQ records given (our 'val' split = CWQ dev; ids are unique across files),
    same format as webqsp_gold_queries; an object MID without a known name becomes '?unmapped'."""
    queries = {}
    for split in ('train', 'dev', 'test'):
        for q in json.load(open(f'{CWQ_JSON}/ComplexWebQuestions_{split}.json')):
            record = records.get(q['ID'])
            if record is None:
                continue
            parsed = chain_and_constraints_from_sparql(parse_sparql(q['sparql']))
            if not parsed['ok']:
                continue
            topic_names = [record['entity_names'][i] for i in record['q_ent']]
            known = {m: mid_to_name[m] for m in [parsed['topic']] + parsed['other_anchors'] if m in mid_to_name}
            unused_topics = [t for t in topic_names if t not in known.values()]
            if parsed['topic'] not in known and len(unused_topics) == 1:
                known[parsed['topic']] = unused_topics[0]
            if parsed['topic'] not in known:
                continue                                       # topic MID without a name
            constraints = []
            for c in parsed['constraints']:
                obj = c['subject'] if c['attaches_to'] and c['object'] in c['attaches_to'] else c['object']
                if FREEBASE_MID.match(obj):
                    obj = known.get(obj, mid_to_name.get(obj, '?unmapped'))
                elif obj.startswith('?'):
                    obj = '?v'
                constraints.append({'predicate': c['predicate'], 'object': obj, 'kind': c['kind']})
            queries[q['ID']] = {'topic': known[parsed['topic']], 'constraints': constraints, 'chain': parsed['chain']}
    return queries


def gold_queries_for(dataset, records):
    """{question id: gold query} for the records of one split (records: {id: record})"""
    webqsp_queries, mid_to_name = webqsp_gold_queries()
    if dataset == 'webqsp':
        return {i: q for i, q in webqsp_queries.items() if i in records}
    return cwq_gold_queries(records, mid_to_name)


# ============================================================ 2. walk a gold query inside a subgraph
def subgraph_triples(record):
    """the record's triples as (head name, relation, tail name), in index order"""
    names, relations = record['entity_names'], record['relation_list']
    return [(names[h], relations[r], names[t]) for h, r, t in
            zip(record['h'].tolist(), record['r'].tolist(), record['t'].tolist())]


def triple_lookup(triples):
    """-> (objects by (subject, predicate), subjects by (predicate, object))"""
    objects, subjects = collections.defaultdict(list), collections.defaultdict(list)
    for h, r, t in triples:
        objects[(h, r)].append(t); subjects[(r, t)].append(h)
    return objects, subjects


def step(node, hop, direction, objects, subjects):
    """triples leaving `node` along one chain hop -> [(next node, triple)]"""
    if direction == 'fwd':
        return [(t, (node, hop['predicate'], t)) for t in objects.get((node, hop['predicate']), [])]
    return [(h, (h, hop['predicate'], node)) for h in subjects.get((hop['predicate'], node), [])]


def walk_one_trail_per_end(query, objects, subjects):
    """Follow the chain from the topic; keep ONE trail per end node (the last one found, as v1 and the eval targets
    were built). -> {end node: [triples]}; empty when the chain breaks"""
    frontier = {query['topic']: []}
    for hop in query['chain']:
        following = {}
        for node, trail in frontier.items():
            for nxt, triple in step(node, hop, hop['direction'], objects, subjects):
                following[nxt] = trail + [triple]
        frontier = following
        if not frontier:
            break
    return frontier


def walk_all_trails(query, objects, subjects):
    """Follow the chain from the topic keeping EVERY trail (an end reached through several mediator nodes keeps all of
    them), at most MAX_TRAILS per hop. -> [(end node, [triples])]; empty when the chain breaks"""
    trails = [(query['topic'], [])]
    for hop in query['chain']:
        trails = [(nxt, trail + [triple]) for node, trail in trails
                  for nxt, triple in step(node, hop, hop['direction'], objects, subjects)][:MAX_TRAILS]
        if not trails:
            break
    return trails


def walk_with_reverse_fallback(query, objects, subjects):
    """As walk_all_trails, but a hop with no triple in its declared direction is tried in the reverse one (Freebase
    keeps many relations as reverse pairs). -> (trails reached, number of hops walked, whether a hop was reversed)"""
    trails, reversed_any = [(query['topic'], [])], False
    for depth, hop in enumerate(query['chain']):
        for direction in (hop['direction'], 'rev' if hop['direction'] == 'fwd' else 'fwd'):
            following = [(nxt, trail + [triple]) for node, trail in trails
                         for nxt, triple in step(node, hop, direction, objects, subjects)]
            if following:
                reversed_any |= direction != hop['direction']
                break
        if not following:
            return trails, depth, reversed_any
        trails = following[:MAX_TRAILS]
    return trails, len(query['chain']), reversed_any


def object_of(constraint):
    return '?v' if constraint['object'] == '?unmapped' else constraint['object']


def constraint_triples(query, nodes, objects, subjects):
    """Triples of the gold constraints touching `nodes`: the predicate matches and the far end equals the constraint
    object ('?v' = any object). Used by v2, v3."""
    found = set()
    for c in query['constraints']:
        obj = object_of(c)
        for node in nodes:
            found |= {(node, c['predicate'], t) for t in objects.get((node, c['predicate']), []) if obj.startswith('?') or obj == t}
            found |= {(h, c['predicate'], node) for h in subjects.get((c['predicate'], node), []) if obj.startswith('?') or obj == h}
    return found


def nodes_of(triples, topic):
    return {topic} | {x for t in triples for x in (t[0], t[2])}


def is_unnamed(node):
    """mediator (CVT) / unnamed entities appear as raw ids in our records"""
    return bool(FREEBASE_MID.match(node)) or node.startswith(('m.', 'g.'))


# ============================================================ 3. the three label versions
def label_v1(query, triples, index):
    """every trail along the gold chain (one per end node) + constraint triples on its nodes.
    v1 constraint rule (kept as built): a triple touching a chain node counts if SOME constraint with its predicate
    has a variable object or names one of the triple's two ends."""
    objects, subjects = triple_lookup(triples)
    frontier = walk_one_trail_per_end(query, objects, subjects)
    chain = {t for trail in frontier.values() for t in trail}
    nodes = set(frontier) | nodes_of(chain, query['topic'])
    touching = set()
    for c in query['constraints']:
        for node in nodes:
            touching |= {(node, c['predicate'], t) for t in objects.get((node, c['predicate']), [])}
            touching |= {(h, c['predicate'], node) for h in subjects.get((c['predicate'], node), [])}
    cons = {t for t in touching - chain
            if any(c['predicate'] == t[1] and (object_of(c).startswith('?') or object_of(c) in (t[0], t[2]))
                   for c in query['constraints'])}
    return {'chain': sorted(index[t] for t in chain if t in index), 'cons': sorted(index[t] for t in cons if t in index)}


def label_v2(query, triples, index, gold_answers):
    """trails along the gold chain that END AT A GOLD ANSWER + their constraint triples; triples only on trails to
    named non-answers -> 'neg'. None when no trail reaches a gold answer."""
    objects, subjects = triple_lookup(triples)
    trails = walk_all_trails(query, objects, subjects)
    to_answers = [trail for end, trail in trails if end in gold_answers]
    if not to_answers:
        return None
    to_others = [trail for end, trail in trails if end not in gold_answers and not is_unnamed(end)]
    chain = {t for trail in to_answers for t in trail}
    cons = constraint_triples(query, nodes_of(chain, query['topic']), objects, subjects) - chain
    neg = {t for trail in to_others for t in trail} - chain - cons
    as_index = lambda ts: sorted(index[t] for t in ts if t in index)
    return {'chain': as_index(chain), 'cons': as_index(cons), 'neg': as_index(neg)}


def shortest_bridges(sources, gold_answers, neighbours):
    """every shortest undirected path (<= MAX_BRIDGE_HOPS) from any source node to the nearest gold answers
    -> (triples on those paths, the source nodes they start from)"""
    distance, parents, layer = {s: 0 for s in sources}, collections.defaultdict(list), list(sources)
    for d in range(1, MAX_BRIDGE_HOPS + 1):
        next_layer = []
        for u in layer:
            for v, triple in neighbours[u]:
                if v not in distance:
                    distance[v] = d; next_layer.append(v)
                if distance[v] == d:
                    parents[v].append((u, triple))
        if gold_answers & set(next_layer):
            break
        layer = next_layer
    reached = [g for g in gold_answers if g in distance and distance[g] > 0]
    if not reached:
        return set(), set()
    bridge, starts, stack, seen = set(), set(), list(reached), set(reached)
    while stack:                                                # walk the parents back to the sources
        v = stack.pop()
        if distance[v] == 0:
            starts.add(v); continue
        for u, triple in parents[v]:
            bridge.add(triple)
            if u not in seen:
                seen.add(u); stack.append(u)
    return bridge, starts


def label_v3_fallback(query, triples, index, gold_answers):
    """For a question v2 cannot label: reversed hops ('flip'), or the gold trails to the deepest node reached plus the
    shortest bridges to a gold answer ('bridge'). None when the chain breaks at hop 1 or no bridge reaches an answer."""
    objects, subjects = triple_lookup(triples)
    trails, depth, _ = walk_with_reverse_fallback(query, objects, subjects)
    to_answers = [trail for end, trail in trails if end in gold_answers] if depth == len(query['chain']) else []
    if to_answers:
        kind, chain = 'flip', {t for trail in to_answers for t in trail}
    elif depth >= 1:
        neighbours = collections.defaultdict(list)
        for triple in dict.fromkeys(triples):
            neighbours[triple[0]].append((triple[2], triple)); neighbours[triple[2]].append((triple[0], triple))
        bridge, starts = shortest_bridges({end for end, _ in trails}, gold_answers, neighbours)
        if not bridge:
            return None
        kind, chain = 'bridge', {t for end, trail in trails if end in starts for t in trail} | bridge
    else:
        return None
    cons = constraint_triples(query, nodes_of(chain, query['topic']), objects, subjects) - chain
    return {'chain': sorted(index[t] for t in chain), 'cons': sorted(index[t] for t in cons), 'neg': [], 'kind': kind}


def build_labels(dataset, split, version):
    """-> ({question id: label}, number of records, number of records with a gold query)"""
    records = {r['id']: r for r in pickle.load(open(f'{DATA}/{dataset}/{split}.pkl', 'rb'))}
    queries = gold_queries_for(dataset, records)
    v2_labels = pickle.load(open(f'{DATA}/{dataset}/sparqlv2_{split}.pkl', 'rb')) if version == 'v3' else {}
    labels = {}
    for qid in (queries if version != 'v3' else records):   # v3 also visits records without a v2 label
        record, query = records[qid], queries.get(qid)
        triples = subgraph_triples(record)
        # a triple listed twice in a record maps to ONE index: v2 / v3 were built with its first index, v1 with its last
        first_index, last_index = {}, {}
        for k, triple in enumerate(triples):
            first_index.setdefault(triple, k); last_index[triple] = k
        index = last_index if version == 'v1' else first_index
        gold_answers = {record['entity_names'][a] for a in record.get('a_ent', [])}
        if version == 'v1':
            if gold_answers:                                    # as built: only questions with an answer in the graph
                labels[qid] = label_v1(query, triples, index)
        elif version == 'v2':
            label = label_v2(query, triples, index, gold_answers)
            if label is not None:
                labels[qid] = label
        else:
            if qid in v2_labels and v2_labels[qid]['chain']:
                labels[qid] = dict(v2_labels[qid], kind='v2'); continue
            if query is None or not gold_answers:
                continue
            label = label_v3_fallback(query, triples, index, gold_answers)
            if label is not None:
                labels[qid] = label
    return labels, len(records), len(queries)


# ============================================================ 4. eval targets (used by retrieve/eval.py)
def gold_query_targets(dataset):
    """Gold-query targets of every TEST question whose gold chain, walked inside its subgraph, reaches a gold answer:
    {question id: (chain triples, constraint triples not on the chain, [trail triples to each gold answer reached])}
    Triples are (head name, relation, tail name). Chain = all trails along the gold chain (one per end node), so it
    includes trails to non-answers; constraint rule as v1 but the object is only checked on (node, predicate, object)
    triples. These are the targets of PathHit / PathAll / Chain-R / Cons-R / Q-F1 in eval.py."""
    records = {r['id']: r for r in pickle.load(open(f'{DATA}/{dataset}/test.pkl', 'rb'))}
    targets = {}
    for qid, query in gold_queries_for(dataset, records).items():
        record = records[qid]
        gold_answers = {record['entity_names'][a] for a in record.get('a_ent', [])}
        if not gold_answers:
            continue
        objects, subjects = triple_lookup(subgraph_triples(record))
        frontier = walk_one_trail_per_end(query, objects, subjects)
        trails_to_answers = [set(trail) for end, trail in frontier.items() if end in gold_answers]
        if not trails_to_answers:
            continue
        chain = {t for trail in frontier.values() for t in trail}
        nodes = set(frontier) | {query['topic']} | {x for trail in frontier.values() for t in trail for x in (t[0], t[2])}
        cons = set()
        for c in query['constraints']:
            obj = object_of(c)
            for node in nodes:
                cons |= {(node, c['predicate'], t) for t in objects.get((node, c['predicate']), [])
                         if not FREEBASE_MID.match(obj) and (obj.startswith('?') or obj == t)}
                cons |= {(h, c['predicate'], node) for h in subjects.get((c['predicate'], node), [])}
        targets[qid] = (chain, cons - chain, trails_to_answers)
    return targets


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description='Build gold-query (SPARQL) training labels for one split.')
    ap.add_argument('--dataset', required=True, choices=['webqsp', 'cwq'])
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--version', required=True, choices=list(LABEL_FILE), help='see the module docstring')
    args = ap.parse_args()
    labels, n_records, n_queries = build_labels(args.dataset, args.split, args.version)
    path = f'{DATA}/{args.dataset}/{LABEL_FILE[args.version]}_{args.split}.pkl'
    pickle.dump(labels, open(path, 'wb'))
    n_with_chain = sum(1 for v in labels.values() if v['chain'])
    kinds = collections.Counter(v.get('kind') for v in labels.values() if v.get('kind'))
    print(f'{args.dataset} {args.split} {args.version}: {n_records} records, {n_queries} with a parsed gold query, '
          f'{n_with_chain} labelled with a non-empty chain (the others train on SP)' + (f', by kind {dict(kinds)}' if kinds else ''))
    print(f'-> {path}')


if __name__ == '__main__':
    main()
