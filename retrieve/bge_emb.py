"""Compute and store BGE-M3 embeddings for SubgraphRAG retriever.

Supports baseline BGE-M3 or any fine-tuned LoRA checkpoint/adapter.
Embeddings are formatted and saved to `data_files/{dataset}/emb/{output_name}/{split}.pth`
in the canonical dictionary format:
    emb_dict[sample_id] = {
        'q_emb': Tensor (1, 1024),
        'entity_embs': Tensor (num_entities, 1024),
        'relation_embs': Tensor (num_relations, 1024),
    }

Usage examples:
    # 1. Baseline BGE-M3 on WebQSP:
    python bge_emb.py -d webqsp --split all --output_name bge-m3-base

    # 2. Fine-tuned LoRA checkpoint:
    python bge_emb.py -d webqsp --split all \
        --adapter data_files/phase11/runs/v1_dense_entity_disjoint/checkpoint-459 \
        --output_name bge-m3-v1
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm


# ============================================================================
# Relation Text Formatting (Self-contained, no external phase11 dependency)
# ============================================================================

def raw_relation(relation: str) -> str:
    return relation


def split_relation(relation: str) -> str:
    return relation.replace(".", " ").replace("_", " ")


def natural_relation(relation: str) -> str:
    parts = relation.split(".")
    if len(parts) >= 3:
        return parts[-1].replace("_", " ")
    return split_relation(relation)


FORMATTERS: dict[str, Callable[[str], str]] = {
    "raw": raw_relation,
    "split": split_relation,
    "natural": natural_relation,
}


def format_relation(relation: str, format_name: str) -> str:
    try:
        formatter = FORMATTERS[format_name]
    except KeyError as error:
        choices = ", ".join(sorted(FORMATTERS))
        raise ValueError(
            f"Unknown relation format {format_name!r}; choose one of: {choices}"
        ) from error
    return formatter(relation)


# ============================================================================
# Path Resolution Helpers
# ============================================================================

def resolve_existing_path(path: str | Path | None) -> Path:
    """Robust path resolver looking in cwd, script dir, and standard repo roots."""
    if path is None:
        return Path(".")
    raw = Path(path)
    if raw.exists():
        return raw.resolve()

    candidates = [
        Path.cwd() / raw,
        Path(__file__).resolve().parent / raw,
        Path(__file__).resolve().parents[1] / raw,
        Path("/home/minhnv/Lab/SubgraphRAG/retrieve") / raw,
        Path("/home/minhnv/Lab/SubgraphRAG") / raw,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return raw


def _validate_cache_tag(cache_tag: str) -> str:
    if cache_tag == "legacy":
        return cache_tag
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", cache_tag):
        raise ValueError(
            "cache_tag must be 'legacy' or a 1-64 character identifier containing "
            "only letters, digits, '.', '_' and '-'"
        )
    return cache_tag


def _cache_dir(data_root: str | Path, dataset: str, kind: str, cache_tag: str) -> Path:
    base = Path(data_root) / dataset / kind / "bge-m3"
    cache_tag = _validate_cache_tag(cache_tag)
    return base if cache_tag == "legacy" else base / cache_tag


def embedding_path(dataset: str, split: str, format_name: str, cache_tag: str = "legacy", data_root: str = "data_files") -> Path:
    return _cache_dir(data_root, dataset, "emb", cache_tag) / f"{split}_{format_name}.pth"


def relation_topic_path(
    dataset: str, split: str, format_name: str, topk: int, cache_tag: str = "legacy", data_root: str = "data_files"
) -> Path:
    return _cache_dir(data_root, dataset, "relation_topics", cache_tag) / f"{split}_{format_name}_top{topk}.pth"


def _hash_file(path: Path, digest: Any) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)


def model_fingerprint(model_name: str | Path) -> str:
    """Return a stable content fingerprint for local checkpoints or Hub identifiers."""
    path = resolve_existing_path(model_name)
    digest = hashlib.sha256()
    if not path.exists():
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(snapshot_download(str(model_name), local_files_only=True))
            revision = snapshot.name
            digest.update(f"hub-revision:{model_name}@{revision}".encode())
            return f"hub-revision-sha256:{digest.hexdigest()}:{revision}"
        except Exception:
            digest.update(f"hub-id:{model_name}".encode())
            return f"hub-id-sha256:{digest.hexdigest()}"

    files = []
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        root = path
        names = {
            "adapter_config.json", "adapter_model.safetensors",
            "adapter_model.bin", "config.json", "model.safetensors",
            "pytorch_model.bin", "colbert_linear.pt", "sparse_linear.pt",
            "m3_heads.pt",
        }
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.name in names)
    if not files:
        raise ValueError(f"Cannot fingerprint model path without recognized files: {path}")
    for file_path in files:
        digest.update(str(file_path.relative_to(root)).encode())
        digest.update(b"\0")
        _hash_file(file_path, digest)
        digest.update(b"\0")
    return f"content-sha256:{digest.hexdigest()}"


def inference_model_fingerprint(model_name: str, adapter_path: str | None = None) -> str:
    base = model_fingerprint(model_name)
    if adapter_path is None:
        return base
    adapter = model_fingerprint(adapter_path)
    digest = hashlib.sha256(f"{base}\n{adapter}\n".encode()).hexdigest()
    return f"base-adapter-sha256:{digest}"


def atomic_torch_save(payload: Any, output: str | Path, overwrite: bool = False) -> None:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file {output}; pass --overwrite explicitly"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def resolve_adapter_artifacts(adapter_path: str | Path | None) -> tuple[Path | None, Path | None]:
    """Resolve a PEFT adapter directory and optional Phase-11 M3 heads."""
    if adapter_path is None:
        return None, None
    raw = resolve_existing_path(adapter_path)
    if not raw.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path} (resolved: {raw})")

    resolved = raw
    if raw.is_dir() and not (raw / "adapter_config.json").is_file():
        nested = raw / "adapter"
        if nested.is_dir() and (nested / "adapter_config.json").is_file():
            resolved = nested

    heads = next(
        (
            candidate
            for candidate in (raw / "m3_heads.pt", resolved.parent / "m3_heads.pt")
            if candidate.is_file()
        ),
        None,
    )
    return resolved, heads


def load_model(model_name: str = "BAAI/bge-m3", adapter_path: str | None = None) -> Any:
    """Load BGE-M3 FlagModel with optional fine-tuned PEFT LoRA adapter."""
    from FlagEmbedding import BGEM3FlagModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE-M3 model '{model_name}' on device '{device}'...")
    # FlagEmbedding changed the constructor keyword from ``device`` to
    # ``devices`` across releases.  Keep this helper compatible with both
    # APIs so analysis can run in either the retriever or BGE environment.
    init_params = inspect.signature(BGEM3FlagModel.__init__).parameters
    model_kwargs = {"use_fp16": device == "cuda"}
    if "devices" in init_params:
        model_kwargs["devices"] = device
    elif "device" in init_params:
        model_kwargs["device"] = device
    model = BGEM3FlagModel(model_name, **model_kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        resolved_adapter, combined_heads = resolve_adapter_artifacts(adapter_path)
        if resolved_adapter is None or not (resolved_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter is incomplete or missing: {resolved_adapter}")
        print(f"Loading LoRA adapter from '{resolved_adapter}'...")
        model.model.model = PeftModel.from_pretrained(
            model.model.model, str(resolved_adapter), is_trainable=False
        )
        if combined_heads is not None:
            states = torch.load(combined_heads, map_location="cpu", weights_only=True)
            if not isinstance(states, dict):
                raise ValueError(f"Malformed M3 heads artifact: {combined_heads}")
            for module_name in ("colbert_linear", "sparse_linear"):
                state = states.get(module_name)
                if not isinstance(state, dict):
                    raise ValueError(f"M3 heads artifact lacks {module_name}: {combined_heads}")
                getattr(model.model, module_name).load_state_dict(state, strict=True)
        adapter_dir = resolved_adapter
        for filename, module_name in (
            ("colbert_linear.pt", "colbert_linear"),
            ("sparse_linear.pt", "sparse_linear"),
        ):
            head_path = adapter_dir / filename
            if combined_heads is None and head_path.exists():
                state = torch.load(head_path, map_location="cpu", weights_only=True)
                getattr(model.model, module_name).load_state_dict(state, strict=True)
        model.model.eval()
    return model


def load_processed_samples(dataset: str, split: str, data_root: Path) -> list[dict[str, Any]]:
    """Load samples from processed split pickle file."""
    path = data_root / dataset / "processed" / f"{split}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset split not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def embed_unique_relations(
    relations: list[str],
    format_name: str,
    model: Any,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Encode all unique relations with BGE-M3 dense representation in batches."""
    if not relations:
        return {}
    texts = [format_relation(rel, format_name) for rel in relations]
    print(f"Encoding {len(relations):,} unique relations (format: '{format_name}')...")
    
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    
    return {
        rel: torch.as_tensor(vec, dtype=torch.float16).cpu()
        for rel, vec in zip(relations, output["dense_vecs"])
    }


def embed_split_to_emb_dir(
    dataset: str,
    split: str,
    output_dir: Path,
    model: Any,
    format_name: str = "split",
    base_emb_name: str = "gte-large-en-v1.5",
    embed_query: bool = True,
    batch_size: int = 64,
    data_root: Path = Path("data_files"),
    overwrite: bool = False,
) -> Path:
    """Build and save full retriever embedding dictionary for a split."""
    final_output = output_dir / f"{split}.pth"
    if final_output.exists() and not overwrite:
        print(f"[skip] {final_output} already exists. Use --overwrite to replace.")
        return final_output

    samples = load_processed_samples(dataset, split, data_root)
    print(f"Loaded {len(samples):,} samples for {dataset}/{split}")

    # 1. Gather all unique relations in this split and batch encode them
    unique_relations = sorted({rel for s in samples for rel in s.get("relation_list", [])})
    rel_emb_map = embed_unique_relations(unique_relations, format_name, model, batch_size)

    # 2. Optionally batch encode questions with BGE-M3
    query_emb_map: dict[str, torch.Tensor] = {}
    if embed_query:
        print(f"Encoding {len(samples):,} queries with BGE-M3...")
        questions = [s["question"] for s in samples]
        q_output = model.encode(
            questions,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        for s, q_vec in zip(samples, q_output["dense_vecs"]):
            query_emb_map[s["id"]] = torch.as_tensor(q_vec, dtype=torch.float16).unsqueeze(0).cpu()

    # 3. Load base embeddings (for entity embeddings and fallback query embeddings)
    base_emb_path = data_root / dataset / "emb" / base_emb_name / f"{split}.pth"
    base_emb_dict: dict[str, Any] = {}
    if base_emb_path.exists():
        print(f"Loading base entity embeddings from '{base_emb_path}'...")
        base_emb_dict = torch.load(base_emb_path, map_location="cpu")
    else:
        print(f"Warning: Base embeddings not found at '{base_emb_path}'. Entity embeddings will be empty.")

    # 4. Construct final canonical embedding bundle for each sample
    print(f"Assembling embedding dictionary for {split}...")
    emb_dict: dict[str, dict[str, torch.Tensor]] = {}
    for s in tqdm(samples, desc=f"Assembling {split}"):
        s_id = s["id"]
        base_sample = base_emb_dict.get(s_id, {})

        # Question embedding
        if embed_query and s_id in query_emb_map:
            q_emb = query_emb_map[s_id]
        elif "q_emb" in base_sample:
            q_emb = base_sample["q_emb"].to(torch.float16)
        else:
            q_emb = torch.zeros(1, 1024, dtype=torch.float16)

        # Entity embeddings from base
        if "entity_embs" in base_sample:
            entity_embs = base_sample["entity_embs"].to(torch.float16)
        else:
            entity_embs = torch.zeros(len(s.get("text_entity_list", [])), 1024, dtype=torch.float16)

        # Relation embeddings from BGE-M3
        sample_rels = s.get("relation_list", [])
        if sample_rels:
            relation_embs = torch.stack([rel_emb_map[rel] for rel in sample_rels])
        else:
            relation_embs = torch.zeros(0, 1024, dtype=torch.float16)

        emb_dict[s_id] = {
            "q_emb": q_emb,
            "entity_embs": entity_embs,
            "relation_embs": relation_embs,
        }

    atomic_torch_save(emb_dict, final_output, overwrite=overwrite)
    print(f" Successfully saved {len(emb_dict):,} sample embeddings to '{final_output}'")
    return final_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode relation (and query) embeddings using BGE-M3 for SubgraphRAG retriever"
    )
    parser.add_argument("-d", "--dataset", choices=["webqsp", "cwq"], default="webqsp")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Base model identifier or local path")
    parser.add_argument(
        "--adapter", default=None, help="Optional path to fine-tuned LoRA checkpoint/adapter directory"
    )
    parser.add_argument("--format", choices=FORMATTERS, default="split", help="Relation text formatter")
    parser.add_argument(
        "--output_name",
        default=None,
        help="Subdirectory name under data_files/{dataset}/emb/{output_name}/ (default: auto-derived)",
    )
    parser.add_argument(
        "--base_emb",
        default="gte-large-en-v1.5",
        help="Base embedding folder to take entity embeddings from (default: gte-large-en-v1.5)",
    )
    parser.add_argument(
        "--embed_query",
        action="store_true",
        default=True,
        help="Also encode query/question text using BGE-M3 (default: True)",
    )
    parser.add_argument(
        "--no_embed_query",
        action="store_false",
        dest="embed_query",
        help="Keep query embedding from base text encoder instead of BGE-M3",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", default="data_files", help="Root data directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    data_root = resolve_existing_path(args.data_root)
    print(f"Data root resolved to: {data_root}")

    # Determine default output name if not provided
    output_name = args.output_name
    if not output_name:
        if args.adapter:
            adapter_stem = Path(args.adapter).name
            parent_stem = Path(args.adapter).parent.name
            output_name = f"bge-m3-{parent_stem}-{adapter_stem}"
        else:
            output_name = "bge-m3-base"

    output_dir = data_root / args.dataset / "emb" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target embedding directory: {output_dir}")

    model = load_model(args.model, args.adapter)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\n--- Processing split: {split} ---")
        embed_split_to_emb_dir(
            dataset=args.dataset,
            split=split,
            output_dir=output_dir,
            model=model,
            format_name=args.format,
            base_emb_name=args.base_emb,
            embed_query=args.embed_query,
            batch_size=args.batch_size,
            data_root=data_root,
            overwrite=args.overwrite,
        )

    print(f"\n Done! To train or infer with standard SubgraphRAG, set in config YAML:\n")
    print(f"dataset:")
    print(f"  name: {args.dataset}")
    print(f"  text_encoder_name: {output_name}")
    print(f"retriever:")
    print(f"  relation_topic_source: gte  # calculates cosine similarity on-the-fly from q_emb and relation_embs\n")


if __name__ == "__main__":
    main()
