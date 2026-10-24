"""Retrieval evaluation (SubgraphRAG's eval.py, extended): one table for any number of retrieval files on one test set.

  python eval.py -d webqsp -p exp_qrdde/runs/webqsp_wl_gte_sparql2_s42/pred_webqsp_base.pth [-p ...] [--subset true]
  python eval.py -d cwq -p WL=path/a.pth -p RBF=path/b.pth --subset false
A retrieval file is {question id: {'scored_triples': [(head, relation, tail, score), ...] best first, ...}}
(exp_qrdde/infer.py, relrag/infer.py, SubgraphRAG's inference.py). '-p name=path' sets the row name.

WHICH QUESTIONS ARE COUNTED (printed above every table, read it before comparing numbers):
  --subset false  every test question (WebQSP 1,639; CWQ 3,531). A question whose answer is not in its subgraph, or
                  that is missing from a file, counts as 0 for answer recall.
  --subset true   only the questions whose answer entity is in the subgraph (WebQSP 1,559; CWQ 2,848): the
                  convention of the SubgraphRAG paper and of every retrieval table in BENCHMARK.md / the report.
  A question missing from a file always counts as 0 (never silently dropped); the coverage line says how many.
  Each metric family is averaged over the questions (of the chosen set) that HAVE its target:
    ansR      answer recall: share of the gold answers (in the subgraph) found as head / tail of the top-k triples
    SP-R/F1   recall / F1 against the shortest-path triples topic -> answer (REL-RAG records, forward direction)
    GPT-R     recall of the GPT-4o evidence triples (SubgraphRAG release)
    gold-query metrics, against the gold SPARQL / InferentialChain walked inside the subgraph (labels.py), on the
    questions whose gold chain reaches a gold answer (WebQSP 1,496; CWQ 2,510):
      PathHit   the top-k contains every triple of at least one gold trail topic -> answer (a complete reasoning path)
      PathAll   ... of the gold trails to every answer reached
      Chain-R   recall of all triples on the gold chain;  Cons-R  recall of the gold-constraint triples (questions
                with >= 1);  Q-F1  F1 of the top-k against chain + constraint triples
  Triples are matched exactly as (head name, relation, tail name).
"""
import argparse, os, pickle, sys
import numpy as np
import torch

ROOT = '/home/minhnv/Lab/SubgraphRAG'
sys.path.insert(0, f'{ROOT}/retrieve/exp_qrdde')
from labels import gold_query_targets

REFERENCE = {   # full test question list + answers in the subgraph (SubgraphRAG release), shortest paths, GPT-4o triples
    'webqsp': dict(answers=f'{ROOT}/reason/scored_triples/webqsp_240912_unidir_test.pth',
                   shortest_paths=f'{ROOT}/relrag/data/test.pkl', gpt=f'{ROOT}/retrieve/data_files/webqsp/gpt_triples.pth'),
    'cwq': dict(answers=f'{ROOT}/reason/scored_triples/cwq_240907_unidir_test.pth',
                shortest_paths=f'{ROOT}/relrag/data/cwq/test.pkl', gpt=f'{ROOT}/reason/gpt_labeled/gpt_labeled_cwq_cleaned.pt'),
}
COLUMNS = (['ansR@50', 'ansR@100', 'ansR@200', 'ansR@500', 
            'PathHit@50', 'PathHit@100', 'PathAll@50', 'Chain-R@50', 'Cons-R@50', 'Q-F1@50'])


def load_targets(dataset):
    """-> (all test question ids, {id: answers in subgraph}, {id: shortest-path triples}, {id: GPT triples},
    {id: gold-query targets})"""
    ref = REFERENCE[dataset]
    answer_file = torch.load(ref['answers'], weights_only=False)
    question_ids = list(answer_file)
    answers = {i: set(v['a_entity_in_graph']) for i, v in answer_file.items() if v['a_entity_in_graph']}
    shortest_paths = {}
    for rec in pickle.load(open(ref['shortest_paths'], 'rb')):
        names, relations = rec['entity_names'], rec['relation_list']
        triples = {(names[rec['h'][k]], relations[rec['r'][k]], names[rec['t'][k]]) for path in rec['sp_paths'] for k in path}
        if triples and rec['id'] in answers:
            shortest_paths[rec['id']] = triples
    gpt_file = torch.load(ref['gpt'], weights_only=False)
    gpt = {i: {tuple(t[:3]) for t in gpt_file[i]} for i in answers if gpt_file.get(i)}
    return question_ids, answers, shortest_paths, gpt, gold_query_targets(dataset)


def f1(true_positives, n_retrieved, n_targets):
    if true_positives == 0:
        return 0.0
    precision, recall = true_positives / n_retrieved, true_positives / n_targets
    return 2 * precision * recall / (precision + recall)


def score_file(retrieved, question_ids, answers, shortest_paths, gpt, gold):
    """mean of every metric over the questions that have its target -> ({column: value in %}, {column: n questions})"""
    values = {c: [] for c in COLUMNS}
    for qid in question_ids:
        ranked = retrieved[qid]['scored_triples'] if qid in retrieved else []      # missing question -> nothing retrieved
        for k in (50, 100, 200, 500):
            top = ranked[:k]
            entities = {x[0] for x in top} | {x[2] for x in top}
            values[f'ansR@{k}'].append(len(answers[qid] & entities) / len(answers[qid]) if qid in answers else 0.0)
        top_triples = {k: {tuple(x[:3]) for x in ranked[:k]} for k in (50, 100)}
        # if qid in shortest_paths:
        #     sp = shortest_paths[qid]
        #     values['SP-R@50'].append(len(sp & top_triples[50]) / len(sp))
        #     values['SP-R@100'].append(len(sp & top_triples[100]) / len(sp))
        #     values['SP-F1@50'].append(f1(len(sp & top_triples[50]), max(len(top_triples[50]), 1), len(sp)))
        # if qid in gpt:
        #     values['GPT-R@50'].append(len(gpt[qid] & top_triples[50]) / len(gpt[qid]))
        #     values['GPT-R@100'].append(len(gpt[qid] & top_triples[100]) / len(gpt[qid]))
        if qid in gold:
            chain, constraints, trails = gold[qid]
            values['PathHit@50'].append(any(t <= top_triples[50] for t in trails))
            values['PathHit@100'].append(any(t <= top_triples[100] for t in trails))
            values['PathAll@50'].append(all(t <= top_triples[50] for t in trails))
            values['Chain-R@50'].append(len(chain & top_triples[50]) / len(chain))
            if constraints:
                values['Cons-R@50'].append(len(constraints & top_triples[50]) / len(constraints))
            query = chain | constraints
            values['Q-F1@50'].append(f1(len(query & top_triples[50]), max(len(top_triples[50]), 1), len(query)))
    return ({c: 100 * float(np.mean(v)) if v else float('nan') for c, v in values.items()},
            {c: len(v) for c, v in values.items()})


def print_table(dataset, subset, n_test, question_ids, rows, counts, coverage):
    family = lambda prefix: counts[next(c for c in COLUMNS if c.startswith(prefix))]
    scope = ('ONLY questions whose answer is in the subgraph (SubgraphRAG convention)' if subset
             else 'FULL test set (answer not in the subgraph -> ansR 0)')
    print(f'\n==== {dataset} test | --subset {str(subset).lower()} | {len(question_ids)} of {n_test} questions: {scope} ====')
    print(f'denominators: ansR {family("ansR")} | gold query: '
          f'PathHit/Chain-R/Q-F1 {family("PathHit")}, Cons-R {family("Cons-R")} (questions with constraint triples)')
    for name, (found, missing) in coverage.items():
        if missing:
            print(f'WARNING: {name}: {missing} of the {len(question_ids)} questions are missing from the file -> counted as 0')
    print('|retriever\t|' + '\t|'.join(COLUMNS) + '\t|')
    print('|---\t|' + '\t|'.join('---' for _ in COLUMNS) + '\t|')
    for name, scores in rows.items():
        print(f'|{name}\t|' + '\t|'.join(f'{scores[c]:.1f}' for c in COLUMNS) + '\t|')


def row_name(spec):
    """'name=path' -> (name, path); a bare path is named '<run directory>/<file name>'"""
    if '=' in spec:
        return spec.split('=', 1)
    return f'{os.path.basename(os.path.dirname(spec))}/{os.path.splitext(os.path.basename(spec))[0]}', spec


def main():
    ap = argparse.ArgumentParser(description='Evaluate retrieval files on one test set (see the module docstring).')
    ap.add_argument('-d', '--dataset', required=True, choices=['webqsp', 'cwq'], help='test set of the files')
    ap.add_argument('-p', '--path', action='append', required=True, help='retrieval file, or name=file; repeatable')
    ap.add_argument('--subset', default='true', choices=['true', 'false'],
                    help='true: only questions whose answer is in the subgraph; false: every test question')
    args = ap.parse_args()
    subset = args.subset == 'true'
    all_ids, answers, shortest_paths, gpt, gold = load_targets(args.dataset)
    question_ids = [i for i in all_ids if i in answers] if subset else all_ids
    rows, coverage, counts = {}, {}, None
    for spec in args.path:
        name, path = row_name(spec)
        retrieved = torch.load(path, weights_only=False)
        found = sum(1 for i in question_ids if i in retrieved)
        coverage[name] = (found, len(question_ids) - found)
        rows[name], counts = score_file(retrieved, question_ids, answers, shortest_paths, gpt, gold)
    print_table(args.dataset, subset, len(all_ids), question_ids, rows, counts, coverage)


if __name__ == '__main__':
    main()
