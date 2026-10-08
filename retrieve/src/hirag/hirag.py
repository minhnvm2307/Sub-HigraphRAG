from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import torch

from src.model.text_encoders.gte_large_en import GTELargeEN

from ._op import (
    build_community_reports,
    build_query_context,
    ingest_sample,
    semantic_cluster_layers,
    structural_communities,
)
from .base import QueryContext, QueryParam
from .prompt import PROMPTS
from .storage import EntityVectorIndex, GraphStorage, JsonKVStorage

logger = logging.getLogger("retrieve.hirag")


def setup_logger(level: str = "INFO") -> None:
    logger_obj = logging.getLogger("retrieve.hirag")
    if not logger_obj.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger_obj.addHandler(handler)
    logger_obj.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger_obj.propagate = False


@dataclass
class HiRAG:
    working_dir: str
    namespace: str = "retrieve_hirag"
    text_encoder: object | None = None
    llm_callable: object | None = None
    semantic_max_layers: int = 3
    semantic_min_cluster_size: int = 3
    semantic_cluster_threshold: float = 0.1
    semantic_stop_sparsity_threshold: float = 0.98
    semantic_stop_sparsity_change_rate: float = 0.05
    semantic_max_length_in_cluster: int = 6000
    community_report_max_input_chars: int = 9000
    meta_entity_types: list[str] = field(default_factory=lambda: PROMPTS["META_ENTITY_TYPES"][:])
    log_level: str = "INFO"
    graph: GraphStorage = field(init=False)
    entities: EntityVectorIndex = field(init=False)
    metadata: JsonKVStorage = field(init=False)
    community_reports: JsonKVStorage = field(init=False)

    def __post_init__(self) -> None:
        setup_logger(self.log_level)
        root = os.path.join(self.working_dir, self.namespace)
        os.makedirs(root, exist_ok=True)
        self.root = root
        logger.info(
            "Initialize HiRAG: root=%s namespace=%s semantic_max_layers=%d semantic_min_cluster_size=%d llm=%s",
            self.root,
            self.namespace,
            self.semantic_max_layers,
            self.semantic_min_cluster_size,
            "enabled" if self.llm_callable is not None else "disabled",
        )
        self.graph = GraphStorage(root)
        self.entities = EntityVectorIndex(root)
        self.metadata = JsonKVStorage(root, "meta")
        self.community_reports = JsonKVStorage(root, "community_reports")

    def index_samples(self, samples: Iterable[dict]) -> dict:
        # ── Giai đoạn 1: Ingest từng sample ─────────────────────────────
        # Kiểm tra xem đã ingest chưa (resume nếu crash giữa chừng)
        ingested_ids = set(self.metadata.get("ingested_sample_ids", []))
        num_samples = len(ingested_ids)
        
        for sample in samples:
            sample_id = str(sample["id"])
            if sample_id in ingested_ids:
                logger.info("Skip already-ingested sample: id=%s", sample_id)
                continue
            
            ingest_sample(self.graph, self.entities, sample)
            ingested_ids.add(sample_id)
            num_samples += 1
            
            # Checkpoint mỗi 10 samples
            if num_samples % 10 == 0:
                self.metadata.update({"ingested_sample_ids": sorted(ingested_ids)})
                self.graph.save()
                self.entities.save()
                self.metadata.save()
                logger.info("Checkpoint saved: samples=%d", num_samples)

        # Lưu toàn bộ sau ingest xong
        self.metadata.update({"ingested_sample_ids": sorted(ingested_ids)})
        self.graph.save()
        self.entities.save()
        self.metadata.save()
        logger.info("Ingest phase complete: total_samples=%d", num_samples)

        # ── Giai đoạn 2: Semantic clustering ────────────────────────────
        if not self.metadata.get("semantic_layers_done"):
            semantic_layers = semantic_cluster_layers(
                self.graph, self.entities,
                embedding_func=self._embedding_func,
                llm_callable=self.llm_callable,
                max_layers=self.semantic_max_layers,
                min_cluster_size=self.semantic_min_cluster_size,
                cluster_threshold=self.semantic_cluster_threshold,
                stop_sparsity_threshold=self.semantic_stop_sparsity_threshold,
                stop_sparsity_change_rate=self.semantic_stop_sparsity_change_rate,
                max_length_in_cluster=self.semantic_max_length_in_cluster,
                meta_entity_types=self.meta_entity_types,
            )
            self.graph.save()
            self.entities.save()
            self.metadata.update({
                "semantic_layers": semantic_layers,
                "semantic_layers_done": True,
            })
            self.metadata.save()
            logger.info("Semantic clustering phase saved to disk")
        else:
            semantic_layers = self.metadata.get("semantic_layers", [])
            logger.info("Skip semantic clustering: already done, layers=%d", len(semantic_layers))

        # ── Giai đoạn 3: Structural communities ─────────────────────────
        if not self.metadata.get("communities_done"):
            communities = structural_communities(self.graph)
            self.metadata.update({
                "communities_raw": communities,  # lưu tạm để resume nếu LLM crash
                "communities_done": True,
            })
            self.metadata.save()
            logger.info("Structural communities phase saved to disk")
        else:
            communities = self.metadata.get("communities_raw", {})
            logger.info("Skip structural communities: already done, communities=%d", len(communities))

        # ── Giai đoạn 4: Community reports (LLM, dễ crash nhất) ─────────
        if not self.metadata.get("reports_done"):
            reports = build_community_reports(
                self.graph, communities,
                llm_callable=self.llm_callable
            )
            self.community_reports.clear()
            self.community_reports.update(reports)
            self.community_reports.save()
            self.metadata.update({"reports_done": True})
            self.metadata.save()
            logger.info("Community reports phase saved to disk")
        else:
            reports = dict(self.community_reports.items())
            logger.info("Skip community reports: already done, reports=%d", len(reports))

        # ── Finalize ─────────────────────────────────────────────────────
        stats = {
            "num_samples": num_samples,
            "num_nodes": self.graph.graph.number_of_nodes(),
            "num_edges": self.graph.graph.number_of_edges(),
            "num_semantic_layers": len(semantic_layers),
            "num_communities": len(reports),
        }
        self.metadata.update({"index_stats": stats})
        self.persist()
        logger.info("Indexing finished: %s", stats)
        return stats

    def reset_index(self, phase: str = "all") -> None:
        """
        phase: "all" | "semantic" | "communities" | "reports"
        Dùng khi muốn rerun 1 giai đoạn cụ thể mà không mất ingest data.
        """
        flags = {
            "all":         ["ingested_sample_ids", "semantic_layers_done", 
                            "communities_done", "reports_done", "index_stats"],
            "semantic":    ["semantic_layers_done", "communities_done", 
                            "reports_done", "index_stats"],
            "communities": ["communities_done", "reports_done", "index_stats"],
            "reports":     ["reports_done", "index_stats"],
        }
        for key in flags.get(phase, []):
            self.metadata.pop(key, None)
        self.metadata.save()
        logger.info("Reset index phase=%s", phase)

    async def query(self, question: str, query_embedding: torch.Tensor, param: QueryParam | None = None) -> QueryContext:
        param = param or QueryParam()
        logger.info("Run query: mode=%s question=%s", param.mode, question[:120])
        return await build_query_context(
            self.graph,
            self.entities,
            self.community_reports,
            question=question,
            query_embedding=query_embedding,
            param=param,
        )

    async def query_sample(self, sample: dict, param: QueryParam | None = None) -> QueryContext:
        logger.info("Run query for sample id=%s", sample.get("id"))
        return await self.query(sample["question"], sample["q_emb"], param=param)

    def query_sync(self, question: str, query_embedding: torch.Tensor, param: QueryParam | None = None) -> QueryContext:
        import asyncio

        return asyncio.run(self.query(question, query_embedding, param=param))

    def embed_query(self, question: str) -> torch.Tensor:
        return self._embedding_func([question])[0]

    def _embedding_func(self, texts: list[str]) -> torch.Tensor:
        if self.text_encoder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.text_encoder = GTELargeEN(device=device)
            logger.info("Initialized text encoder for query embedding on device=%s", device)
        return self.text_encoder.embed(texts)

    def is_indexed(self) -> bool:
        stats = self.metadata.get("index_stats", {})
        return bool(stats.get("num_samples", 0) > 0)

    def load_or_index(self, samples, force_reindex=False):
        if not force_reindex and self.is_indexed():
            logger.info("Cache found, skipping indexing: %s", self.summary())
            return self.metadata.get("index_stats", {})
        return self.index_samples(samples)

    def persist(self) -> None:
        logger.info("Persist HiRAG artifacts to disk")
        self.graph.save()
        self.entities.save()
        self.metadata.save()
        self.community_reports.save()

    def summary(self) -> dict:
        stats = self.metadata.get("index_stats", {})
        return {"working_dir": self.root, **stats}

    def asdict(self) -> dict:
        return {
            "working_dir": self.root,
            "namespace": self.namespace,
            "semantic_max_layers": self.semantic_max_layers,
            "semantic_min_cluster_size": self.semantic_min_cluster_size,
            "summary": self.summary(),
        }
