from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import torch

from retrieve.src.model.text_encoders.gte_large_en import GTELargeEN

from ._op import (
    build_community_reports,
    build_query_context,
    ingest_sample,
    semantic_cluster_layers,
    structural_communities,
)
from .base import QueryContext, QueryParam
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
    semantic_max_layers: int = 3
    semantic_min_cluster_size: int = 3
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
            "Initialize HiRAG: root=%s namespace=%s semantic_max_layers=%d semantic_min_cluster_size=%d",
            self.root,
            self.namespace,
            self.semantic_max_layers,
            self.semantic_min_cluster_size,
        )
        self.graph = GraphStorage(root)
        self.entities = EntityVectorIndex(root)
        self.metadata = JsonKVStorage(root, "meta")
        self.community_reports = JsonKVStorage(root, "community_reports")

    def index_samples(self, samples: Iterable[dict]) -> dict:
        logger.info("Start indexing samples")
        num_samples = 0
        for sample in samples:
            ingest_sample(self.graph, self.entities, sample)
            num_samples += 1
            if num_samples % 10 == 0:
                logger.info(
                    "Indexing progress: samples=%d nodes=%d edges=%d",
                    num_samples,
                    self.graph.graph.number_of_nodes(),
                    self.graph.graph.number_of_edges(),
                )

        semantic_layers = semantic_cluster_layers(
            self.graph,
            self.entities,
            max_layers=self.semantic_max_layers,
            min_cluster_size=self.semantic_min_cluster_size,
        )
        communities = structural_communities(self.graph)
        reports = build_community_reports(self.graph, communities)

        self.community_reports.clear()
        self.community_reports.update(reports)
        self.metadata.update(
            {
                "index_stats": {
                    "num_samples": num_samples,
                    "num_nodes": self.graph.graph.number_of_nodes(),
                    "num_edges": self.graph.graph.number_of_edges(),
                    "num_semantic_layers": len(semantic_layers),
                    "num_communities": len(reports),
                },
                "semantic_layers": semantic_layers,
            }
        )
        self.persist()
        stats = self.metadata.get("index_stats", {})
        logger.info("Indexing finished: %s", stats)
        return stats

    def query(self, question: str, query_embedding: torch.Tensor, param: QueryParam | None = None) -> QueryContext:
        param = param or QueryParam()
        logger.info("Run query: mode=%s question=%s", param.mode, question[:120])
        return build_query_context(
            self.graph,
            self.entities,
            self.community_reports,
            question=question,
            query_embedding=query_embedding,
            param=param,
        )

    def query_sample(self, sample: dict, param: QueryParam | None = None) -> QueryContext:
        logger.info("Run query for sample id=%s", sample.get("id"))
        return self.query(sample["question"], sample["q_emb"], param=param)

    def embed_query(self, question: str) -> torch.Tensor:
        if self.text_encoder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.text_encoder = GTELargeEN(device=device)
            logger.info("Initialized text encoder for query embedding on device=%s", device)
        return self.text_encoder.embed([question])[0]

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
