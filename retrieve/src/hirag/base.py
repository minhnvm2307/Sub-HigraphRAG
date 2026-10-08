from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QueryParam:
    mode: Literal["hi", "hi_local", "hi_global", "hi_bridge"] = "hi"
    top_k: int = 8
    top_m: int = 3
    level: int = 0
    max_token_for_community_report: int = 1800
    max_token_for_text_unit: int = 1800
    max_token_for_local_context: int = 1800
    max_token_for_bridge_knowledge: int = 1800
    community_single_one: bool = False
    only_need_context: bool = True
    response_type: str = "short"
    use_query_scores: bool = True
    # Backward-compatible knobs used by the previous pipeline.
    top_k_entities: int = 8
    top_k_triples: int = 20
    top_k_communities: int = 5
    max_bridge_paths: int = 5
    max_bridge_hops: int = 4
    community_level: int = 0
    # Scoring method: 'similarity' or 'mlp'
    scoring_method: Literal["similarity", "mlp", "hybrid"] = "similarity"
    # Optional path to precomputed MLP results .pth file
    mlp_result_path: str | None = None
    sample_id: str | None = None
    mlp_top_k: int = 50
    # Weight applied to MLP-derived per-entity scores when merging with similarity
    mlp_weight: float = 1.0


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
    context: str | None = None
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    reasoning_path: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    # Backward-compatible fields.
    entity_hits: list[EntityHit] = field(default_factory=list)
    local_triples: list[dict] = field(default_factory=list)
    bridge_paths: list[dict] = field(default_factory=list)
    community_reports: list[CommunityReport] = field(default_factory=list)

