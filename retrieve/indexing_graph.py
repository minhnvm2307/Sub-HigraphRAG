from __future__ import annotations

import hashlib
import json
import os
import re
from argparse import ArgumentParser
import asyncio
from dataclasses import asdict
import requests
import dotenv

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset
from src.hirag.hirag import HiRAG, setup_logger

dotenv.load_dotenv()  # Load environment variables from .env file

FPT_KEY=os.getenv("FPT_KEY")
FPT_BASE_URL=os.getenv("FPT_BASE_URL")

def ollama_chat(messages, model="llama3.1"):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "llama3.1",
        "messages": messages,
        "stream": False,
        "keep_alive": "10m"
    }

    response = requests.post(url, json=payload)
    return response.json()

def gpt_chat(messages, model="Llama-3.3-70B-Instruct"):
    from openai import OpenAI

    client = OpenAI(api_key=FPT_KEY, base_url=FPT_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def llm_infer(message, llm_model="fpt"):
    if "ollama" in llm_model:
        outputs = ollama_chat(messages=[{"role": "assistant", "content": message}])
        return outputs["message"]["content"]
    elif "fpt" in llm_model:
        outputs = gpt_chat(messages=[{"role": "assistant", "content": message}])
    else:
        print(f"Unsupported LLM model: {llm_model}, defaulting to FPT")
        outputs = gpt_chat(messages=[{"role": "assistant", "content": message}])
    return outputs


def _slugify_question(question: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", question.lower()).strip("_")
    if not slug:
        slug = "question"
    return slug[:max_length].strip("_") or "question"


def _graph_signature(sample: dict) -> str:
    triples = zip(
        sample.get("h_id_list", []),
        sample.get("r_id_list", []),
        sample.get("t_id_list", []),
    )
    canonical = sorted((str(h_id), str(r_id), str(t_id)) for h_id, r_id, t_id in triples)
    payload = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _question_identity(sample: dict) -> str:
    sample_id = str(sample.get("id", "unknown"))
    question = sample.get("question", "")
    return f"{sample_id}_{_slugify_question(question)}"


def _build_question_graph_map(samples: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    sample_map: dict[str, dict] = {}

    for sample in samples:
        sample_id = str(sample.get("id", "unknown"))
        question = sample.get("question", "")
        graph_hash = _graph_signature(sample)
        folder_name = _question_identity(sample)

        group = grouped.setdefault(
            graph_hash,
            {
                "graph_hash": graph_hash,
                "sample_ids": [],
                "question_identities": [],
                "questions": [],
            },
        )
        group["sample_ids"].append(sample_id)
        group["question_identities"].append(folder_name)
        group["questions"].append(question)

        sample_map[sample_id] = {
            "sample_id": sample_id,
            "question": question,
            "graph_hash": graph_hash,
            "folder_name": folder_name,
        }

    return {
        "num_samples": len(samples),
        "num_groups": len(grouped),
        "groups": grouped,
        "samples": sample_map,
    }


async def main(args) -> None:
    setup_logger(args.log_level)

    config_file = args.config_file or f"configs/retriever/{args.dataset}.yaml"
    config = load_yaml(config_file)

    dataset = RetrieverDataset(
        config=config,
        split=args.split,
        skip_no_path=args.skip_no_path,
    )
    samples = dataset.processed_dict_list
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
    if not samples:
        raise ValueError("No samples available for testing")

    namespace = args.namespace or f"{args.dataset}_{args.split}"
    index_root = os.path.join(args.working_dir, namespace)
    os.makedirs(index_root, exist_ok=True)

    question_graph_map = _build_question_graph_map(samples)
    mapping_file = os.path.join(index_root, "question_graph_map.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(question_graph_map, f, indent=2, ensure_ascii=True)

    print(
        f"Saved question-graph mapping first: {mapping_file} "
        f"(samples={question_graph_map['num_samples']}, groups={question_graph_map['num_groups']})"
    )

    indexed_count = 0
    skipped_count = 0
    for sample in samples:
        sample_id = str(sample.get("id", "unknown"))
        sample_meta = question_graph_map["samples"][sample_id]
        sample_namespace = sample_meta["folder_name"]
        
        sample_hirag = HiRAG(
            working_dir=index_root,
            namespace=sample_namespace,
            llm_callable=lambda prompt: llm_infer(prompt, llm_model=args.llm_model),
            semantic_max_layers=args.semantic_max_layers,
            semantic_min_cluster_size=args.semantic_min_cluster_size,
            log_level=args.log_level,
        )

        if not args.force_reindex and sample_hirag.is_indexed():
            skipped_count += 1
            print(f"Skip already indexed sample {sample_id}: {sample_namespace}")
            continue

        sample_stats = sample_hirag.index_samples([sample])
        indexed_count += 1
        print(
            f"Indexed sample {sample_id} -> {sample_namespace} | "
            f"nodes={sample_stats.get('num_nodes', 0)} edges={sample_stats.get('num_edges', 0)}"
        )

    print(
        f"\nArtifacts written to: {index_root} | indexed={indexed_count} skipped={skipped_count}"
    )



if __name__ == "__main__":
    parser = ArgumentParser("HiRAG small-sample debug runner")
    parser.add_argument("--dataset", type=str, default="webqsp", choices=["cwq", "webqsp"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default="indexed_hirag")
    parser.add_argument("--namespace", type=str, default="webqsp_subset")
    parser.add_argument("--num_samples", type=int, default=0, help="0 means index all samples")
    parser.add_argument("--semantic_max_layers", type=int, default=3)
    parser.add_argument("--semantic_min_cluster_size", type=int, default=3)
    parser.add_argument("--llm_model", type=str, default="ollama:llama")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--skip_no_path", action="store_true")
    parser.add_argument("--force_reindex", action="store_true", help="Ignore cached index and rebuild all per-sample graphs")
    asyncio.run(main(parser.parse_args()))
    # message = "Question: Which rivers flow through the city of Paris? Answer:"
    # outputs = llm_infer(message)
    # print(outputs)
