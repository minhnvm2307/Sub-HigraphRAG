from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QueryParam:
    mode: Literal["hi", "hi_local", "hi_global", "hi_bridge"] = "hi"
    top_k_entities: int = 8
    top_k_triples: int = 20
    top_k_communities: int = 5
    max_bridge_paths: int = 5
    max_bridge_hops: int = 4
    community_level: int = 0


@dataclass
class EntityHit:
    node_id: str
    entity_name: str
    score: float
    node_data: dict = field(default_factory=dict)


@dataclass
class CommunityReport:
    community_id: str
    level: int
    title: str
    summary: str
    node_ids: list[str]
    edge_ids: list[str]
    rating: float
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryContext:
    question: str
    mode: str
    entity_hits: list[EntityHit] = field(default_factory=list)
    local_triples: list[dict] = field(default_factory=list)
    bridge_paths: list[dict] = field(default_factory=list)
    community_reports: list[CommunityReport] = field(default_factory=list)

