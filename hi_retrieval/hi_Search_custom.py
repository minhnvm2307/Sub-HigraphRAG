"""
HiRAG with custom LLM provider support.

Supported providers (set PROVIDER variable below):
  - "gemini"   : Google Gemini (native API)
  - "ollama"   : Local Ollama models
  - "custom"   : Any OpenAI-compatible API (FPT AI, Together, Groq, Mistral, etc.)

Usage:
    Edit config.yaml to fill in your provider credentials, then set PROVIDER below.
"""

import os
import logging
import numpy as np
import yaml
from dataclasses import dataclass
from openai import AsyncOpenAI

from hirag import HiRAG, QueryParam
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from hirag._storage import Neo4jStorage, NetworkXStorage
from sentence_transformers import SentenceTransformer

# ── Choose your provider ──────────────────────────────────────────────────────
PROVIDER = "custom"   # "gemini" | "ollama" | "custom"
# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

hirag_cfg = config["hirag"]


# ── Embedding wrapper (same pattern as existing hi_Search_*.py files) ─────────

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    def decorator(func) -> EmbeddingFunc:
        return EmbeddingFunc(**kwargs, func=func)
    return decorator

def _build_ollama():
    cfg = config["ollama"]
    base_url = cfg["base_url"]  # e.g. http://localhost:11434

    @wrap_embedding_func_with_attrs(
        embedding_dim=cfg.get("embedding_dim", 768),
        max_token_size=config["model_params"]["max_token_size"],
    )
    async def ollama_embedding(texts: list[str]) -> np.ndarray:
        # Ollama exposes an OpenAI-compatible /v1 endpoint
        client = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        response = await client.embeddings.create(
            model=cfg["embedding_model"], input=texts
        )
        return np.array([dp.embedding for dp in response.data])

    async def ollama_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        client = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        model_name = cfg["model"]
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        if hashing_kv is not None:
            args_hash = compute_args_hash(model_name, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.chat.completions.create(
            model=model_name, messages=messages, **kwargs
        )
        result = response.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": model_name}}
            )
        return result

    return ollama_embedding, ollama_model_if_cache


# ── Provider: Custom OpenAI-compatible (FPT AI, Together, Groq, Mistral…) ────

def _build_custom():
    cfg = config["custom"]
    api_key = cfg.get("api_key") or os.getenv("CUSTOM_API_KEY")
    base_url = cfg["base_url"]
    model_name = cfg["model"]

    # If no dedicated embedding model, fall back to GLM embeddings
    embed_model = cfg.get("embedding_model") or ""
    if embed_model:
        sentence_transformer_model = SentenceTransformer(
            embed_model,
            tokenizer_kwargs={"clean_up_tokenization_spaces": True},
        )
        inferred_embed_dim = sentence_transformer_model.get_sentence_embedding_dimension()
        cfg_embed_dim = cfg.get("embedding_dim")
        if cfg_embed_dim is not None and cfg_embed_dim != inferred_embed_dim:
            logging.warning(
                "custom.embedding_dim (%s) mismatches model output dim (%s). Using model dim.",
                cfg_embed_dim,
                inferred_embed_dim,
            )

        @wrap_embedding_func_with_attrs(
            embedding_dim=inferred_embed_dim,
            max_token_size=config["model_params"]["max_token_size"],
        )
        async def custom_embedding(texts: list[str]) -> np.ndarray:
            return sentence_transformer_model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
    else:
        raise ValueError(
            "No custom.embedding_model configured. Please set custom.embedding_model in config.yaml"
        )

    async def custom_model_if_cache(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        if hashing_kv is not None:
            args_hash = compute_args_hash(model_name, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        try:
            response = await client.chat.completions.create(
                model=model_name, messages=messages, **kwargs
            )
        except Exception as e:
            logging.error(e)
            return "ERROR"

        result = response.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": model_name}}
            )
        return result

    return custom_embedding, custom_model_if_cache


# ── Wire up the selected provider ─────────────────────────────────────────────

_builders = {
    "ollama": _build_ollama,
    "custom": _build_custom,
}

if PROVIDER not in _builders:
    raise ValueError(f"Unknown PROVIDER '{PROVIDER}'. Choose from: {list(_builders)}")

embedding_func, model_func = _builders[PROVIDER]()

graph_func = HiRAG(
    working_dir=hirag_cfg["working_dir"],
    enable_llm_cache=hirag_cfg["enable_llm_cache"],
    embedding_func=embedding_func,
    best_model_func=model_func,
    cheap_model_func=model_func,
    enable_hierachical_mode=hirag_cfg["enable_hierachical_mode"],
    embedding_batch_num=hirag_cfg["embedding_batch_num"],
    embedding_func_max_async=hirag_cfg["embedding_func_max_async"],
    enable_naive_rag=hirag_cfg["enable_naive_rag"],
    graph_storage_cls=NetworkXStorage,
)

# Comment out after first run (index already built)
with open("index.txt") as f:
    graph_func.insert(f.read())

print("Perform hi search:")
print(graph_func.query("How to treat Infectious Diseases?", param=QueryParam(mode="hi")))
