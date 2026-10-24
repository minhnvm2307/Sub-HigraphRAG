import os
import json
import wandb
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data, get_data_from_custom_retrieval
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all, parse_response

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    elif 'custom' in llm_mode:
        from prompts import custom_sys_prompt, custom_cot_prompt
        return custom_sys_prompt, custom_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")
    file_handle.flush()


# Every test question with its gold answers (SubgraphRAG release): WebQSP 1,639, CWQ 3,531. 'a_entity_in_graph' marks
# the questions of the usual subset (answer in the subgraph): WebQSP 1,559, CWQ 2,848.
TEST_REFERENCE = {"webqsp": "./scored_triples/webqsp_240912_unidir_test.pth",
                  "cwq": "./scored_triples/cwq_240907_unidir_test.pth"}


def full_test_predictions(pred_file_path, dataset_name):
    """Predictions padded to the FULL test set: a test question without a prediction (missing from the retrieval file,
    e.g. a subset-only export, or the WebQSP question without a subgraph) gets the answer "ans: none", i.e. counts as
    wrong.
    Written next to the predictions as <name>-fulltest-predictions.jsonl, so --subset false always means every test
    question."""
    import torch
    reference = torch.load(TEST_REFERENCE[dataset_name], weights_only=False)
    rows = [json.loads(line) for line in open(pred_file_path)]
    answered = {row["id"] for row in rows}
    missing = [qid for qid in reference if qid not in answered]
    # "ans: none", not "": both evals read it as "no answer", while an empty string is a substring of every gold answer
    # and would count as a hit in the original eval's double-check branch (when / where / language ... questions)
    rows += [{"id": qid, "question": reference[qid]["question"], "prediction": "ans: none", "a_entity": reference[qid]["a_entity"],
              "supports": []} for qid in missing]
    padded_path = Path(str(pred_file_path).replace("-predictions.jsonl", "-fulltest-predictions.jsonl"))
    with open(padded_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[full test set] {len(answered)} of {len(reference)} test questions have a prediction; "
          f"{len(missing)} without one are scored as wrong (answer 'ans: none')")
    return padded_path


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1):

    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}  ->  "
          + ("questions whose answer is in the subgraph only" if subset else "EVERY test question (missing = wrong)"))

    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    if run is not None:
        run.log({f"results{postfix}/hit@1": hit1,
                 f"results{postfix}/macro_f1": f1,
                 f"results{postfix}/macro_precision": prec,
                 f"results{postfix}/macro_recall": recall,
                 f"results{postfix}/exact_match": em,
                 f"results{postfix}/totally_wrong": tw,
                 f"results{postfix}/micro_f1": mi_f1,
                 f"results{postfix}/micro_precision": mi_prec,
                 f"results{postfix}/micro_recall": mi_recall,
                 f"results{postfix}/total_cnt": total_cnt,
                 f"results{postfix}/no_ans_cnt": no_ans_cnt,
                 f"results{postfix}/no_ans_ratio": no_ans_ratio,
                 f"results{postfix}/hal_score": hal_score})  # score_h in the paper
        if stats is not None:
            for k, v in stats.items():
                run.log({f"stats{postfix}/{k}": v})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    if run is not None:
        run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str, help="Path to pre-computed triple scores (scored_triples.pth)")
    parser.add_argument("--custom_retrieval", type=str, help="Path to custom retrieval results (retrieval_result.pth)")
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="ollama:llama", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    parser.add_argument("--ollama_url", type=str, default=None, help="Ollama base URL (e.g. http://localhost:11434)")
    parser.add_argument("--pred_format", choices=["ans", "raw"], default="ans", help="raw = full LLM response as in the original SubgraphRAG code (needed to compare with the papers)")
    parser.add_argument("--workers", type=int, default=1, help="concurrent LLM requests (start ollama with OLLAMA_NUM_PARALLEL >= workers)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="predictions .jsonl of an earlier run with the SAME prompt / model: its answers are reused "
                             "for the questions it covers, only the other questions are sent to the LLM")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    # Determine if using custom retrieval or pre-processed RoG results
    use_custom_retrieval = args.custom_retrieval is not None
    
    if not use_custom_retrieval:
        pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}"
    if use_custom_retrieval:
        run_name += "-custom_retrieval"
    try:
        run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)
    except Exception as e:
        print(f"[Warning] Failed to initialize wandb ({e}). Continuing without wandb tracking.")
        run = None

    if args.score_dict_path is None and not use_custom_retrieval:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    if use_custom_retrieval:
        c_path = Path(args.custom_retrieval)
        custom_name = c_path.parent.name if c_path.stem == "retrieval_result" else f"{c_path.parent.name}_{c_path.stem}"
        raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-custom_{custom_name}-predictions.jsonl"

    else:
        raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions.jsonl"

    if args.pred_format == "raw":
        raw_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.name.replace("-predictions", "-raw-predictions"))
    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty, ollama_base_url=args.ollama_url)
    
    # Load data based on source
    if use_custom_retrieval:
        print(f"Loading custom retrieval results from: {args.custom_retrieval}")
        data = get_data_from_custom_retrieval(dataset_name, args.custom_retrieval, split, prompt_mode)
    else:
        data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)
    
    if split == "test":
        import torch
        n_test = len(torch.load(TEST_REFERENCE[dataset_name], weights_only=False))
        print(f"[coverage] the retrieval covers {len(data)} of the {n_test} {dataset_name} test questions"
              + ("" if len(data) == n_test else f"; the {n_test - len(data)} others are scored as wrong in the full evaluation"))
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Limited evaluation to {len(data)} samples.")

    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    print("Starting inference...")
    done_ids = {row["id"] for row in load_checkpoint(raw_pred_file_path)}
    if args.resume_from:                                  # copy the answers of an earlier run for the questions we have
        wanted = {qa["id"] for qa in data} - done_ids
        reused = [row for row in load_checkpoint(args.resume_from) if row["id"] in wanted]
        with open(raw_pred_file_path, "a") as pred_file:
            for row in reused:
                save_checkpoint(pred_file, row)
        done_ids |= {row["id"] for row in reused}
        print(f"Reused {len(reused)} answers from {args.resume_from}")
    data = [qa for qa in data if qa["id"] not in done_ids]   # only the questions without an answer yet
    start_idx = 0
    print(f"Already answered: {len(done_ids)}; to send to the LLM: {len(data)}")

    if start_idx >= len(data):
        print("All items have already been processed. Exiting.")
    else:
        def finish(each_qa, res):
            del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]
            print(f"RESPONSE: {res}")
            raw = res[0] if isinstance(res, (list, tuple)) and res else res
            res, supports = parse_response(res)
            if args.pred_format == "raw":
                # original SubgraphRAG (0db77b5) stores the full response: its Hit metric searches the whole text, so the
                # numbers in the SubgraphRAG/REL-RAG papers are only comparable with raw predictions
                res = raw
            # each_qa["prediction"] = res[0]
            each_qa["prediction"] = res
            each_qa["supports"] = supports
            return {k: each_qa[k] for k in ["id", "question", "prediction", "a_entity", "supports"]}

        with open(raw_pred_file_path, "a") as pred_file:
            if args.workers <= 1:
                for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
                    res = llm_inf_all(llm, each_qa, llm_mode, model_name)
                    save_checkpoint(pred_file, finish(each_qa, res))
            else:
                # Concurrent requests (the ollama server batches them when started with OLLAMA_NUM_PARALLEL >= workers).
                # Sliding window: the workers always have requests queued, while results are written in input order,
                # so one long generation delays only the writing and the line-count resume above stays valid.
                from collections import deque
                from concurrent.futures import ThreadPoolExecutor
                todo = iter(data[start_idx:])

                def safe_inf(qa):
                    # one failed request must not abort the whole run: retry, then record an empty answer
                    for attempt in range(3):
                        try:
                            return llm_inf_all(llm, qa, llm_mode, model_name)
                        except Exception as e:
                            if 'budget reached' in str(e):
                                raise  # stop the run (resumable) instead of writing empty answers past the budget
                            print(f"[{qa['id']}] attempt {attempt + 1} failed: {e!r}")
                    return ["", ""]
                window = deque()
                with ThreadPoolExecutor(args.workers) as ex, tqdm(initial=start_idx, total=len(data)) as bar:
                    def submit():
                        qa = next(todo, None)
                        if qa is not None:
                            window.append((qa, ex.submit(safe_inf, qa)))
                    for _ in range(8 * args.workers):
                        submit()
                    while window:
                        each_qa, fut = window.popleft()
                        save_checkpoint(pred_file, finish(each_qa, fut.result()))
                        bar.update(1)
                        submit()

    # If the processing completes, rename the files to remove the "resume" flag
    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    if raw_pred_file_path != final_pred_file_path:
        os.rename(raw_pred_file_path, final_pred_file_path)
    eval_all(final_pred_file_path, run, subset=True)
    if split == "test":
        eval_all(full_test_predictions(final_pred_file_path, dataset_name), run, subset=False)
    else:
        eval_all(final_pred_file_path, run, subset=False)



if __name__ == "__main__":
    main()
    # import json
    # with open("./results/KGQA/webqsp/HiRAG/ollama:llama/scored_100_hi_bridge-sys_icl_dc-0-split_test-n30-predictions.jsonl", "r") as f:
    #     # pretty save in new file (filter some keys)
    #     data = [json.loads(line) for line in f]
    #     remained_keys = ["id", "question", "prediction", "a_entity", "supports"]
    #     for item in data:
    #         keys_to_remove = set(item.keys()) - set(remained_keys)
    #         for key in keys_to_remove:
    #             del item[key]
    # with open("./results/KGQA/webqsp/HiRAG/ollama:llama/scored_100_hi_bridge-sys_icl_dc-0-split_test-n30-predictions_pretty.json", "w") as f:
    #     json.dump(data, f, indent=2)