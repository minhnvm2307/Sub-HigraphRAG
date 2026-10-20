"""Step 7 — score the generated answers. The RAPL release has no scoring script and the paper does not say how Hit was
computed, so three readings are reported side by side (SQuAD-style normalisation: lowercase, no punctuation/articles):
  saved   Hit = a gold answer appears anywhere in the saved file. The authors' saved file holds prompt + answer
          (HF generate output), so this counts answers that were only in the retrieved paths -> an upper bound, NOT QA.
  answer  Hit = a gold answer appears in the model's completion only (text after the final "assistant" turn).
  parsed  Hit / Macro-F1 on the line "The answer is: ..." of the completion (the authors' template); items split on
          , ; / | and " and ". F1 per question: precision = predicted items matching a gold answer, recall = gold
          answers matched by a predicted item (match = either normalised string contains the other).
Every CWQ test question counts (3,531); a question with no answer file scores 0. Sample i = question i of
data_files/cwq/processed/test_retrieval.pkl, i.e. of the SubgraphRAG/RoG test order (checked against each sample file).
  python rapl_faithful/score_answers.py --file_dir <model_inference output dir>
"""
import argparse, json, os, pickle, re, string
import numpy as np

ROOT = os.environ.get('SGRAG_ROOT', '/home/minhnv/Lab/SubgraphRAG')


def norm(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def match(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_dir', required=True)
    ap.add_argument('--retrieval_pkl', default=f'{ROOT}/RAPL/data_files/cwq/processed/test_retrieval.pkl')
    args = ap.parse_args()
    if not os.path.exists(args.retrieval_pkl):          # e.g. scoring on a machine without the RAPL build: the
        args.retrieval_pkl = f'{ROOT}/retrieve/data_files/cwq/processed/test.pkl'   # same order (checked in step 1b)
    recs = pickle.load(open(args.retrieval_pkl, 'rb'))
    gold_src = {r['id']: r for r in pickle.load(open(f'{ROOT}/retrieve/data_files/cwq/processed/test.pkl', 'rb'))}
    ans_dir = os.path.join(args.file_dir, 'llama31_8b_generated_answers')
    rows, missing, q_mismatch = [], 0, 0
    for i, s in enumerate(recs):
        gold = list(dict.fromkeys(gold_src[s['id']]['a_entity']))
        sf = os.path.join(args.file_dir, f'sample_{i}.txt')
        if os.path.exists(sf) and s['question'] not in open(sf, encoding='utf-8').read():
            q_mismatch += 1
        af = os.path.join(ans_dir, f'sample_{i}.txt')
        if not os.path.exists(af):
            missing += 1
            rows.append((0, 0, 0, 0.0)); continue
        saved = open(af, encoding='utf-8').read()
        completion = saved.split('assistant\n\n')[-1] if 'assistant\n\n' in saved else saved
        m = re.search(r'the answer is\s*:?\s*(.+)', completion, flags=re.I)
        pred = m.group(1).strip() if m else ''
        items = [x for x in re.split(r',|;|/|\||\band\b', pred) if norm(x)]
        hit_saved = any(norm(g) and norm(g) in norm(saved) for g in gold)
        hit_ans = any(norm(g) and norm(g) in norm(completion) for g in gold)
        hit_parsed = any(match(g, pred) for g in gold) if pred else False
        if items and gold:
            p = np.mean([any(match(x, g) for g in gold) for x in items])
            r = np.mean([any(match(x, g) for x in items) for g in gold])
            f1 = 2 * p * r / (p + r) if p + r else 0.0
        else:
            f1 = 0.0
        rows.append((hit_saved, hit_ans, hit_parsed, f1))
    a = np.array(rows, dtype=float)
    res = {'n_questions': len(recs), 'answer_files_missing': missing, 'question_text_mismatch': q_mismatch,
           'Hit_saved_text (prompt+answer, upper bound)': round(a[:, 0].mean() * 100, 2),
           'Hit_answer_only': round(a[:, 1].mean() * 100, 2),
           'Hit_parsed': round(a[:, 2].mean() * 100, 2), 'MacroF1_parsed': round(a[:, 3].mean() * 100, 2)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(os.path.join(args.file_dir, 'scores.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
