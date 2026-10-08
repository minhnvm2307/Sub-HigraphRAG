from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

REASON_DIR = Path(__file__).resolve().parent
REPO_ROOT = REASON_DIR.parent
RETRIEVE_SRC = REPO_ROOT / "retrieve"

if str(REASON_DIR) not in sys.path:
    sys.path.insert(0, str(REASON_DIR))
if str(RETRIEVE_SRC) not in sys.path:
    sys.path.insert(0, str(RETRIEVE_SRC))

import llm_utils as llm_utils_module
from llm_utils import llm_init, llm_inf_all

from preprocess.prepare_data import get_data_from_custom_retrieval, get_data
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original
from src.hirag.hirag import HiRAG, QueryParam, setup_logger
from src.model.text_encoders.gte_large_en import GTELargeEN


GRAPH_ICL_USER_PROMPT = """Retrieved graph evidence:
Local retrieved triples:
- (Paris, country, France) [score=0.99]
- (Paris, capital_of, France) [score=0.97]

Relevant community report:
- Community 12: Paris is connected to France through location and capital relations.

Question:
What country is Paris in?"""

GRAPH_ICL_ASS_PROMPT = "ans: France"

llm_utils_module.icl_user_prompt = GRAPH_ICL_USER_PROMPT
llm_utils_module.icl_ass_prompt = GRAPH_ICL_ASS_PROMPT


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        ckpt = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ckpt.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed checkpoint line in {file_path}")
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
    print(f"Evaluating on subset: {subset}")

    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(
        str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops
    )
    postfix = "_sub" if subset else ""
    run.log(
        {
            f"results{postfix}/hit@1": hit1,
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
            f"results{postfix}/hal_score": hal_score,
        }
    )
    if stats is not None:
        for key, value in stats.items():
            run.log({f"stats{postfix}/{key}": value})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)


def _truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_triple(triple) -> tuple[str, str, str, float | None]:
    head = str(triple[0])
    relation = str(triple[1])
    tail = str(triple[2])
    score = None
    if len(triple) > 3:
        try:
            score = float(triple[3])
        except (TypeError, ValueError):
            score = None
    return head, relation, tail, score


def _extract_seed_entities(triples: list[tuple]) -> list[str]:
    seeds = []
    for triple in triples:
        head, _, tail, _ = _normalize_triple(triple)
        seeds.append(head)
        seeds.append(tail)
    return _unique_preserve_order([seed for seed in seeds if seed])


def _format_scored_triples(triples: list[tuple], max_triples: int) -> str:
    lines = []
    for triple in triples[:max_triples]:
        head, relation, tail, score = _normalize_triple(triple)
        score_suffix = f" [score={score:.4f}]" if score is not None else ""
        lines.append(f"- ({head}, {relation}, {tail}){score_suffix}")
    return "\n".join(lines)


def _rank_community_reports(
    community_store,
    seed_entities: list[str],
    top_k: int,
    max_level: int | None = None,
) -> list[dict]:
    seed_set = set(seed_entities)
    scored_reports: list[dict] = []

    for community_id, report in community_store.items():
        if max_level is not None and int(report.get("level", 0)) > max_level:
            continue
        node_ids = report.get("node_ids", [])
        overlap = len(seed_set.intersection(node_ids))
        if overlap <= 0:
            continue
        ranked_report = dict(report)
        ranked_report["overlap"] = overlap
        ranked_report["rank_score"] = overlap + float(report.get("rating", 0.0)) * 0.05
        scored_reports.append(ranked_report)

    if not scored_reports:
        for _, report in community_store.items():
            ranked_report = dict(report)
            ranked_report["overlap"] = 0
            ranked_report["rank_score"] = float(report.get("rating", 0.0))
            scored_reports.append(ranked_report)

    scored_reports.sort(key=lambda item: (item.get("rank_score", 0.0), item.get("rating", 0.0)), reverse=True)
    return scored_reports[:top_k]


def _build_bridge_paths(graph_storage, seed_entities: list[str], max_paths: int, max_hops: int) -> list[list[str]]:
    if max_paths <= 0:
        return []

    bridge_paths = []
    seen_paths = set()
    eligible_entities = [entity for entity in seed_entities if graph_storage.graph.has_node(entity)]

    for idx, source in enumerate(eligible_entities):
        for target in eligible_entities[idx + 1 :]:
            for path in graph_storage.shortest_paths(source, target, cutoff=max_hops):
                key = tuple(path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                bridge_paths.append(path)
                if len(bridge_paths) >= max_paths:
                    return bridge_paths
    return bridge_paths


def _format_local_triples(triples: list[tuple]) -> str:
    lines = []
    for triple in triples:
        source, relation, target, score = _normalize_triple(triple)
        score_suffix = f" [score={score:.4f}]" if score is not None else ""
        lines.append(f"- {source} --{relation}--> {target}{score_suffix}")
    return "\n".join(lines)


def _format_bridge_paths(bridge_paths: list[list[str]]) -> str:
    lines = []
    for path in bridge_paths:
        hops = max(len(path) - 1, 0)
        lines.append(f"- {' -> '.join(path)} [hops={hops}]")
    return "\n".join(lines)


def _format_community_reports(community_reports: list[dict]) -> str:
    lines = []
    for report in community_reports:
        summary = _truncate_text(report.get("summary", ""), 260)
        lines.append(
            f"- community={report.get('community_id')} [overlap={report.get('overlap', 0)}, rating={float(report.get('rating', 0.0)):.2f}] "
            f"{report.get('title', '')}: {summary}"
        )
    return "\n".join(lines)

def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"
score_dict_path = "./scored_triples/retrieval_result.pth"
global_data = get_data_from_custom_retrieval("webqsp", score_dict_path, "test", "icl_100")

def build_hirag_prompts(sample: dict, query_context) -> dict:
    global global_data
    max_sections = 3
    max_lines_per_section = 12
    max_chars_per_section = 16000

    def _format_relations(relations: list[dict], limit: int) -> str:
        lines = []
        for rel in relations[:limit]:
            src, tgt = rel.get("src_tgt", ("", ""))
            relation = rel.get("relation", "related_to")
            score = rel.get("rank", None)
            score_suffix = f" [score={score:.4f}]" if isinstance(score, (int, float)) else ""
            # lines.append(f"- ({src}, {relation}, {tgt}){score_suffix}")
            lines.append(f"- ({src}, {relation}, {tgt})")
        return "\n".join(lines)

    def _format_reasoning_path(paths: list[dict], limit: int) -> str:
        lines = []
        for item in paths[:limit]:
            src, tgt = item.get("src_tgt", ("", ""))
            relation = item.get("relation", "related_to")
            lines.append(f"( {src} --{relation}--> {tgt} )")
        return "\n".join(lines)

    def _format_communities(communities: list[dict], limit: int) -> str:
        lines = []
        for report in communities[:limit]:
            summary = _truncate_text(report.get("summary", ""), 220)
            title = report.get("title", "")
            rating = float(report.get("rating", 0.0))
            # lines.append(f"- {title} [rating={rating:.2f}]: {summary}")
            lines.append(f"- {title}: {summary}")
        return "\n".join(lines)

    sections = []

    if query_context.mode in {"hi", "hi_bridge"} and query_context.reasoning_path:
        path_text = _format_reasoning_path(query_context.reasoning_path, max_lines_per_section)
        if path_text:
            sections.append("Bridge reasoning edges:\n" + _truncate_text(path_text, max_chars_per_section))

    if query_context.mode in {"hi", "hi_global"} and query_context.communities:
        comm_text = _format_communities(query_context.communities, max_lines_per_section)
        if comm_text:
            sections.append("Community reports:\n" + _truncate_text(comm_text, max_chars_per_section))

    # if query_context.relations:
    #     rel_text = _format_relations(query_context.relations, max_lines_per_section)
    #     if rel_text:
    #         sections.append("Top retrieved triples:\n" + _truncate_text(rel_text, max_chars_per_section))
    # score_dict_path = "./scored_triples/retrieval_result.pth"
    # data = get_data_from_custom_retrieval("webqsp", score_dict_path, "test", "icl_100")
    
    # triples_of_sample = []
    # for item in global_data:
    #     if item["id"] == sample["id"]:
    #         triples_of_sample = item["scored_triplets"]
    #         if not triples_of_sample:
    #             print(f"ERROR: No scored triples found for sample ID {sample['id']} in custom retrieval results.")
    #         break
    # triples_of_sample = [(triplet[0], triplet[1], triplet[2]) for triplet in triples_of_sample]
    # triples_of_sample = triples_of_sample[:100]
    
    # sections.append("Top retrieved triples:\n" + "\n".join([triplet_to_str(triplet) for triplet in triples_of_sample]))

    evidence_text = "\n\n".join(sections[:max_sections]) if sections else "No graph evidence was retrieved."

    print(f"Constructed evidence for question '{sample['question']}':\n{evidence_text[:1500]}...") 
    sys_prompt = (
        "Answer the question using only the retrieved graph evidence from SubgraphRAG and the indexed graph. "
        'Return every valid answer on a separate line, each prefixed with "ans:". '
        'There could be multiple valid answers, return all of them with "ans: answer1\nans: answer2". '
        'If the evidence is insufficient, return "ans: not available".'
    )
    cot_prompt = (
        "Format the final answer as a concise list. "
        'Use one answer per line with prefix "ans:". '
        'The answer should exactly match the entity name in the given evidence if applicable. '
        "Remove duplicates and do not add explanations."
    )
    user_prompt = "\n\n".join(
        [
            f"Question:\n{sample['question']}",
            f"Retrieved graph evidence (mode: {query_context.mode}):\n{evidence_text}",
        ]
    )
    return {
        "sys_query": sys_prompt,
        "user_query": user_prompt,
        "cot_query": cot_prompt,
    }


def load_question_samples(dataset_name: str, split: str, num_samples: int, local_retrieval_path: str | None) -> list[dict]:
    if local_retrieval_path:
        retrieval_path = Path(local_retrieval_path)
        if not retrieval_path.exists():
            raise FileNotFoundError(f"Retrieval result not found: {retrieval_path}")

        samples = get_data_from_custom_retrieval(dataset_name, str(retrieval_path), split, prompt_mode="scored_100")
        print(f"Loading {num_samples}/{len(samples)} samples from custom retrieval results at {retrieval_path}")
    else:
        raise ValueError("Local retrieval path is required to load question samples with graph context. Please provide --local_retrieval_path.")

    if num_samples > 0:
        samples = samples[:num_samples]
    return samples


def load_question_graph_map(index_root: Path) -> dict:
    mapping_file = index_root / "question_graph_map.json"
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Missing question graph map: {mapping_file}. "
            "Run retrieve/indexing_graph.py first to build per-question graph folders."
        )
    with mapping_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def route_samples_to_index(samples: list[dict], sample_graph_map: dict) -> tuple[list[tuple[dict, dict]], list[str]]:
    question_lookup = {}
    for sample_id, meta in sample_graph_map.items():
        question = meta.get("question", "")
        if question and question not in question_lookup:
            question_lookup[question] = meta

    routed = []
    unmatched = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        meta = sample_graph_map.get(sample_id)
        if meta is None:
            meta = question_lookup.get(sample.get("question", ""))
        if meta is None:
            unmatched.append(sample_id or sample.get("question", "<unknown>"))
            continue
        routed.append((sample, meta))
    return routed, unmatched


def get_query_embedding(sample: dict, shared_encoder: GTELargeEN | None = None) -> torch.Tensor:
    query_emb = sample.get("q_emb")
    if query_emb is None:
        if shared_encoder is None:
            raise ValueError(
                "Sample is missing q_emb and no shared GPU query encoder is available. "
                "Ensure retrieval_result contains q_emb or run with CUDA available."
            )
        return shared_encoder.embed([sample["question"]])[0]
    if isinstance(query_emb, torch.Tensor):
        return query_emb.float()
    return torch.tensor(query_emb, dtype=torch.float32)


def main(args) -> None:
    os.chdir(REASON_DIR)
    setup_logger(args.log_level)

    if args.index_dir:
        index_path = Path(args.index_dir).expanduser().resolve()
        working_dir = str(index_path.parent)
        namespace = index_path.name
    else:
        working_dir = args.working_dir
        namespace = args.namespace or f"{args.dataset_name}_{args.split}"

    dataset_name = args.dataset_name
    split = args.split
    query_mode = args.query_mode
    llm_mode = args.llm_mode
    model_name = args.model_name

    run_name = f"{model_name}-{query_mode}-{llm_mode}-{args.frequency_penalty}-{split}"
    if args.num_samples > 0:
        run_name += f"-n{args.num_samples}"

    run = wandb.init(project=f"HiRAG-{dataset_name}", name=run_name, config=args)

    llm = llm_init(
        model_name,
        args.tensor_parallel_size,
        args.max_seq_len_to_capture,
        args.max_tokens,
        args.seed,
        args.temperature,
        args.frequency_penalty,
    )

    shared_query_encoder = GTELargeEN(device="cuda")
    print("Initialized shared query encoder on device=cuda")

    index_root = Path(working_dir) / namespace
    if not index_root.exists():
        raise ValueError(f"Index root does not exist: {index_root}")

    question_graph_map = load_question_graph_map(index_root)
    sample_graph_map = question_graph_map.get("samples", {})
    if not sample_graph_map:
        raise ValueError(
            f"No sample mappings found in {index_root / 'question_graph_map.json'}. "
            "Re-run indexing_graph.py to generate per-sample graph folders."
        )

    print("Loaded question-graph mapping:")
    print(
        json.dumps(
            {
                "index_root": str(index_root),
                "num_samples_in_map": question_graph_map.get("num_samples", 0),
                "num_graph_groups": question_graph_map.get("num_groups", 0),
            },
            indent=2,
        )
    )

    samples = load_question_samples(dataset_name, split, args.num_samples, args.local_retrieval_path)
    if not samples:
        raise ValueError("No samples available for testing")

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/HiRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    sample_tag = f"-n{args.num_samples}" if args.num_samples > 0 else ""
    raw_pred_file_path = raw_pred_folder_path / (
        f"scored_10_{query_mode}-{llm_mode}-{args.frequency_penalty}-split_{split}{sample_tag}-predictions.jsonl"
    )

    print("Generating prompts...")
    print(f"Query mode: {query_mode}")
    print(f"Index root: {index_root}")
    if args.local_retrieval_path:
        print(f"Local retrieval path: {args.local_retrieval_path}")

    routed_samples, unmatched_samples = route_samples_to_index(samples, sample_graph_map)
    if unmatched_samples:
        print(
            f"Warning: {len(unmatched_samples)} samples do not match indexed graphs and will be skipped. "
            f"Examples: {unmatched_samples[:5]}"
        )
    if not routed_samples:
        raise ValueError("No samples matched the indexed graph folders.")

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))
    print(f"Resuming from index: {start_idx}")

    if start_idx >= len(routed_samples):
        print("All items have already been processed. Exiting.")
    else:
        hirag_cache: dict[str, HiRAG] = {}
        with open(raw_pred_file_path, "a", encoding="utf-8") as pred_file:
            for each_qa, sample_meta in tqdm(
                routed_samples[start_idx:],
                initial=start_idx,
                total=len(routed_samples),
                desc="Inference",
                unit="sample",
            ):
                sample_namespace = sample_meta["folder_name"]
                if sample_namespace not in hirag_cache:
                    sample_hirag = HiRAG(
                        working_dir=str(index_root),
                        namespace=sample_namespace,
                        llm_callable=llm,
                        text_encoder=shared_query_encoder,
                        log_level=args.log_level,
                    )
                    if not sample_hirag.is_indexed():
                        raise ValueError(
                            f"Sample index is missing or incomplete: {sample_hirag.root}"
                        )
                    hirag_cache[sample_namespace] = sample_hirag

                sample_hirag = hirag_cache[sample_namespace]
                query_embedding = get_query_embedding(each_qa, shared_query_encoder)
                query_param = QueryParam(
                    mode=query_mode,
                    top_k=args.top_k_entities,
                    top_m=args.top_m,
                    level=args.community_level,
                    top_k_triples=args.top_k_triples,
                    top_k_communities=args.top_k_communities,
                    max_bridge_paths=args.max_bridge_paths,
                    max_bridge_hops=args.max_bridge_hops,
                    use_query_scores=not args.disable_query_scores,
                    scoring_method="hybrid",
                    mlp_result_path=args.mlp_result_path,
                    sample_id=str(each_qa.get("id")),
                )
                query_context = sample_hirag.query_sync(
                    each_qa["question"],
                    query_embedding=query_embedding,
                    param=query_param,
                )
                prompts = build_hirag_prompts(each_qa, query_context)
                res = llm_inf_all(llm, prompts, llm_mode, model_name)

                each_qa["prediction"] = str(res[0]).strip()
                each_qa["used_index_namespace"] = sample_namespace
                save_checkpoint(pred_file, each_qa)

    eval_all(raw_pred_file_path, run, subset=True)
    # eval_all(raw_pred_file_path, run, subset=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiRAG reasoning runner")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--index_dir", type=str, default=None, help="Path to a pre-built HiRAG index root")
    parser.add_argument("--working_dir", type=str, default="../retrieve/indexed_hirag", help="HiRAG working directory")
    parser.add_argument("--namespace", type=str, default="index_graph", help="HiRAG namespace under working_dir")
    parser.add_argument("--local_retrieval_path", type=str, default="./scored_triples/retrieval_result.pth", help="Path to SubgraphRAG MLP retrieval_result.pth")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit the number of samples for debugging")
    parser.add_argument("--query_mode", type=str, default="hi_global", choices=["hi", "hi_local", "hi_global", "hi_bridge"], help="HiRAG retrieval mode")
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="ollama:llama", help="Model name")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0, help="Frequency penalty")
    parser.add_argument("--top_k_entities", type=int, default=8, help="Top-k entity hits")
    parser.add_argument("--top_m", type=int, default=3, help="Top-m key entities for guided paths")
    parser.add_argument("--top_k_triples", type=int, default=20, help="Top-k local triples")
    parser.add_argument("--top_k_communities", type=int, default=5, help="Top-k community reports")
    parser.add_argument("--max_bridge_paths", type=int, default=5, help="Max bridge paths")
    parser.add_argument("--max_bridge_hops", type=int, default=4, help="Max bridge hops")
    parser.add_argument("--community_level", type=int, default=0, help="Community level")
    parser.add_argument("--mlp_result_path", type=str, default="./scored_triples/retrieval_result.pth", help="Path to precomputed MLP retrieval_result.pth")
    parser.add_argument("--disable_query_scores", action="store_true", help="Disable query-aware scoring")
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    main(parser.parse_args())
