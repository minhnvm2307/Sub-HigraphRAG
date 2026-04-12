from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import torch
import torch.nn.functional as F

logger = logging.getLogger("retrieve.hirag")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_load(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_dump(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


@dataclass
class JsonKVStorage:
    working_dir: str
    namespace: str
    _data: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_dir(self.working_dir)
        self.path = os.path.join(self.working_dir, f"kv_store_{self.namespace}.json")
        self._data = _json_load(self.path, {})
        logger.info(
            "KV storage ready: namespace=%s path=%s entries=%d",
            self.namespace,
            self.path,
            len(self._data),
        )

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def update(self, values: dict) -> None:
        self._data.update(values)

    def clear(self) -> None:
        self._data = {}

    def save(self) -> None:
        _json_dump(self.path, self._data)
        logger.info(
            "KV storage saved: namespace=%s path=%s entries=%d",
            self.namespace,
            self.path,
            len(self._data),
        )


@dataclass
class EntityVectorIndex:
    working_dir: str
    namespace: str = "entities"
    ids: list[str] = field(init=False, default_factory=list)
    embeddings: torch.Tensor = field(init=False, default_factory=lambda: torch.empty(0))
    meta: dict[str, dict] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_dir(self.working_dir)
        self.path = os.path.join(self.working_dir, f"vdb_{self.namespace}.pt")
        if os.path.exists(self.path):
            payload = torch.load(self.path, map_location="cpu")
            self.ids = payload["ids"]
            self.embeddings = payload["embeddings"].float()
            self.meta = payload["meta"]
        logger.info(
            "Vector index ready: namespace=%s path=%s entities=%d",
            self.namespace,
            self.path,
            len(self.ids),
        )

    def upsert(self, records: Iterable[tuple[str, torch.Tensor, dict]]) -> None:
        emb_map = {node_id: self.embeddings[i] for i, node_id in enumerate(self.ids)}
        num_records = 0
        for node_id, embedding, metadata in records:
            if embedding.ndim != 1:
                embedding = embedding.reshape(-1)
            emb_map[node_id] = embedding.detach().cpu().float()
            self.meta[node_id] = metadata
            num_records += 1
        self.ids = sorted(emb_map)
        if self.ids:
            self.embeddings = torch.stack([emb_map[node_id] for node_id in self.ids], dim=0)
        else:
            self.embeddings = torch.empty(0)
        logger.info(
            "Vector index upserted: namespace=%s records=%d total_entities=%d",
            self.namespace,
            num_records,
            len(self.ids),
        )

    def query(self, query_embedding: torch.Tensor, top_k: int) -> list[dict]:
        if not self.ids:
            return []
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        scores = F.cosine_similarity(
            F.normalize(self.embeddings, dim=1),
            F.normalize(query_embedding.reshape(1, -1).float(), dim=1),
        )
        top_k = min(top_k, len(self.ids))
        values, indices = torch.topk(scores, top_k)
        return [
            {
                "id": self.ids[index],
                "score": float(score),
                "metadata": self.meta.get(self.ids[index], {}),
            }
            for score, index in zip(values.tolist(), indices.tolist())
        ]

    def save(self) -> None:
        torch.save(
            {"ids": self.ids, "embeddings": self.embeddings, "meta": self.meta},
            self.path,
        )
        logger.info(
            "Vector index saved: namespace=%s path=%s entities=%d",
            self.namespace,
            self.path,
            len(self.ids),
        )


@dataclass
class GraphStorage:
    working_dir: str
    namespace: str = "chunk_entity_relation"

    def __post_init__(self) -> None:
        _ensure_dir(self.working_dir)
        self.path = os.path.join(self.working_dir, f"graph_{self.namespace}.graphml")
        if os.path.exists(self.path):
            self.graph = nx.read_graphml(self.path)
        else:
            self.graph = nx.Graph()
        logger.info(
            "Graph storage ready: namespace=%s path=%s nodes=%d edges=%d",
            self.namespace,
            self.path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def save(self) -> None:
        nx.write_graphml(self.graph, self.path)
        logger.info(
            "Graph storage saved: namespace=%s path=%s nodes=%d edges=%d",
            self.namespace,
            self.path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def upsert_node(self, node_id: str, **attrs) -> None:
        current = self.graph.nodes.get(node_id, {})
        merged = {**current, **attrs}
        self.graph.add_node(node_id, **merged)

    def upsert_edge(self, source: str, target: str, **attrs) -> None:
        current = self.graph.edges.get((source, target), {})
        merged = {**current, **attrs}
        self.graph.add_edge(source, target, **merged)

    def node_data(self, node_id: str) -> dict:
        return dict(self.graph.nodes[node_id])

    def edge_data(self, source: str, target: str) -> dict:
        return dict(self.graph.edges[source, target])

    def neighbors_with_edges(self, node_id: str) -> list[dict]:
        rows = []
        if not self.graph.has_node(node_id):
            return rows
        for neighbor in self.graph.neighbors(node_id):
            data = dict(self.graph.edges[node_id, neighbor])
            rows.append({"source": node_id, "target": neighbor, **data})
        return rows

    def shortest_paths(self, source: str, target: str, cutoff: int) -> list[list[str]]:
        if source == target or not self.graph.has_node(source) or not self.graph.has_node(target):
            return []
        try:
            paths = list(nx.all_shortest_paths(self.graph, source=source, target=target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return [path for path in paths if len(path) - 1 <= cutoff]
