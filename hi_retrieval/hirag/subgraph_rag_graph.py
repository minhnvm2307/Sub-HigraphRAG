"""
hirag/subgraph_rag_graph.py

Strategy: Group questions by their q_entity (topic entity).
Questions sharing the same topic entity → merged into 1 graph → 1 .graphml file.

File layout under working_dir/
  graph_<dataset>_<split>_<safe_entity>.graphml   ← one per topic entity group
  graph_index.json                                ← maps q_id → topic_entity

Design decisions
----------------
* A question with multiple q_entities is assigned to the FIRST one (primary topic).
  Change PRIMARY_ENTITY_STRATEGY = "all" to duplicate across every group.
* Edge weight = max retriever score across all questions containing that edge.
* Nodes carry source_id = "<q_id><SEP><q_id>..." so community_schema() works.
* All tuning knobs live in GraphLoadConfig — change there, not inside functions.
* Cached .graphml is reused automatically; delete file to force reload.

Debug tips
----------
* Run as __main__ to smoke-test the first 3 groups.
* Set log level DEBUG to see per-triple detail.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import torch

from ._storage import NetworkXStorage
from ._utils import logger

# ---------------------------------------------------------------------------
# Strategy knob
# "first" → question assigned to q_entity[0] only  (recommended)
# "all"   → question duplicated into every q_entity group
# ---------------------------------------------------------------------------
PRIMARY_ENTITY_STRATEGY = "first"


# ---------------------------------------------------------------------------
# Config — all tunable knobs in one place
# ---------------------------------------------------------------------------

@dataclass
class GraphLoadConfig:
    """
    retrieval_dir        : folder containing the .pth file
    default_entity_type  : entity_type written on every node
    merge_duplicate_edges: keep max-score edge when (h,t) appears >1 in a group
    skip_self_loops      : drop triples where head == tail
    min_triples_to_warn  : log WARNING when a question has fewer triples
    score_field_index    : position of the score in each tuple
                           scored_triples = [(h, r, t, score)] → index 3
                           set None if no score field
    fallback_weight      : edge weight used when score is absent or unparsable
    """
    retrieval_dir: str = "reason/scored_triples"
    default_entity_type: str = "Entity"
    merge_duplicate_edges: bool = True
    skip_self_loops: bool = True
    min_triples_to_warn: int = 1
    score_field_index: Optional[int] = 3
    fallback_weight: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_name(entity: str) -> str:
    """Turn an entity string into a safe filename component (max 80 chars)."""
    return re.sub(r"[^\w\-]", "_", entity)[:80]


def _resolve_pth(retrieval_dir: str, retrieval_file: str) -> Path:
    p = Path(retrieval_dir) / retrieval_file
    if not p.exists():
        raise FileNotFoundError(
            f"[SubgraphRAGGraph] File not found: {p.resolve()}\n"
            f"  → Fix: set GraphLoadConfig.retrieval_dir correctly."
        )
    return p


def _load_pth(path: Path) -> dict:
    """Load and spot-check the .pth dict. Returns raw dict."""
    logger.info(f"[Loader] Reading {path} …")
    raw = torch.load(path, weights_only=False)

    if not isinstance(raw, dict):
        raise TypeError(
            f"[Loader] Expected dict, got {type(raw).__name__} in {path}."
        )

    # Spot-check first entry
    first_key = next(iter(raw))
    first_val = raw[first_key]
    missing = [k for k in ("scored_triples", "q_entity") if k not in first_val]
    if missing:
        raise KeyError(
            f"[Loader] Missing keys {missing} in first entry (q_id={first_key}).\n"
            f"  Available keys: {list(first_val.keys())}"
        )

    logger.info(
        f"[Loader] {len(raw)} questions loaded from {path.name}. "
        f"Sample keys: {list(first_val.keys())}"
    )
    return raw


def _group_by_q_entity(
    raw: dict,
    strategy: str,
) -> Dict[str, List[str]]:
    """
    Returns { topic_entity: [q_id, ...] }.
    Logs size distribution for quick sanity check.
    """
    groups: Dict[str, List[str]] = {}
    skipped = 0

    for q_id, data in raw.items():
        entities: list = data.get("q_entity", [])
        if not entities:
            logger.debug(f"  q_id={q_id} has no q_entity — skipped")
            skipped += 1
            continue
        targets = entities if strategy == "all" else [entities[0]]
        for entity in targets:
            groups.setdefault(entity, []).append(q_id)

    if skipped:
        logger.warning(f"[Grouping] {skipped} questions had no q_entity and were skipped.")

    sizes = sorted(len(v) for v in groups.values())
    logger.info(
        f"[Grouping] {len(groups)} groups from {len(raw)} questions "
        f"(strategy='{strategy}') | "
        f"questions/group: min={sizes[0]}, max={sizes[-1]}, "
        f"median={sizes[len(sizes)//2]}"
    )
    return groups


def _build_group_graph(
    q_ids: List[str],
    raw: dict,
    cfg: GraphLoadConfig,
    label: str,          # just for logging
) -> nx.Graph:
    """
    Merge scored_triples from all q_ids into one nx.Graph.

    Node attrs : entity_type, description, source_id (SEP-joined q_ids)
    Edge attrs : description (relation), weight (max score), source_id
    """
    G = nx.Graph()
    n_self_loops = n_bad = 0

    for q_id in q_ids:
        triples = raw[q_id].get("scored_triples", [])

        if len(triples) < cfg.min_triples_to_warn:
            logger.warning(
                f"  [{label}] q_id={q_id} has only {len(triples)} triple(s)"
            )

        for triple in triples:
            if len(triple) < 3:
                logger.debug(f"  [{label}] malformed triple {triple} in {q_id}")
                n_bad += 1
                continue

            head, relation, tail = str(triple[0]), str(triple[1]), str(triple[2])

            # Score
            weight = cfg.fallback_weight
            if cfg.score_field_index is not None and len(triple) > cfg.score_field_index:
                try:
                    weight = float(triple[cfg.score_field_index])
                except (ValueError, TypeError):
                    pass

            if cfg.skip_self_loops and head == tail:
                n_self_loops += 1
                continue

            # Nodes
            for entity in (head, tail):
                if not G.has_node(entity):
                    G.add_node(
                        entity,
                        entity_type=cfg.default_entity_type,
                        description=entity,
                        source_id=q_id,
                    )
                else:
                    src = G.nodes[entity].get("source_id", "")
                    if q_id not in src:
                        G.nodes[entity]["source_id"] = (
                            f"{src}<SEP>{q_id}" if src else q_id
                        )

            # Edge
            if G.has_edge(head, tail):
                if cfg.merge_duplicate_edges:
                    old_w = G.edges[head, tail].get("weight", cfg.fallback_weight)
                    G.edges[head, tail]["weight"] = max(old_w, weight)
                    old_r = G.edges[head, tail].get("description", "")
                    if relation not in old_r:
                        G.edges[head, tail]["description"] = (
                            f"{old_r} | {relation}" if old_r else relation
                        )
            else:
                G.add_edge(head, tail,
                           description=relation,
                           weight=weight,
                           source_id=q_id)

    logger.info(
        f"  [{label}] graph built — "
        f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"self_loops_skipped={n_self_loops}, bad_triples={n_bad}, "
        f"questions_merged={len(q_ids)}"
    )
    return G


def _patch_global_config(global_config: dict, dataset_name: str, split: str) -> dict:
    """Ensure required keys exist in global_config before passing to NetworkXStorage."""
    cfg = dict(global_config)  # don't mutate caller's dict
    if not cfg.get("working_dir"):
        default = f"./hirag_cache_{dataset_name}_{split}"
        logger.warning(f"[Config] 'working_dir' missing — defaulting to '{default}'")
        cfg["working_dir"] = default
    os.makedirs(cfg["working_dir"], exist_ok=True)
    cfg.setdefault("max_graph_cluster_size", 10)
    cfg.setdefault("graph_cluster_seed", 0xDEADBEEF)
    return cfg


# ---------------------------------------------------------------------------
# SubgraphRAGGraph  — one instance = one topic-entity group
# ---------------------------------------------------------------------------

class SubgraphRAGGraph(NetworkXStorage):
    """
    NetworkXStorage subclass representing a single topic-entity group's graph.

    Always create via the factory methods:

        # All groups at once
        graphs = SubgraphRAGGraph.load_all_groups(
            retrieval_file="retrieval_result.pth",
            dataset_name="webqsp",
            split="test",
            global_config=global_config,
        )

        # Single group (e.g. for resuming)
        graph = SubgraphRAGGraph.load_group(
            topic_entity="Jamaica",
            q_ids=["WebQTest-0", ...],
            raw=raw_pth_dict,
            ...
        )

    After clustering and community reports, persist with:
        await graph.index_done_callback()
    """

    # ------------------------------------------------------------------ #
    # Internal constructor                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, topic_entity: str, dataset_name: str, split: str, global_config: dict):
        global_config = _patch_global_config(global_config, dataset_name, split)

        self.namespace = f"{dataset_name}_{split}_{_safe_name(topic_entity)}"
        self.global_config = global_config
        self.embedding_func = None          # not needed for graph-only indexing

        super().__post_init__()             # NetworkXStorage.__post_init__

        self.topic_entity = topic_entity
        self.dataset_name = dataset_name
        self.split = split

    # ------------------------------------------------------------------ #
    # Factory: single group                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_group(
        cls,
        topic_entity: str,
        q_ids: List[str],
        raw: dict,
        dataset_name: str,
        split: str,
        global_config: dict,
        load_cfg: Optional[GraphLoadConfig] = None,
    ) -> "SubgraphRAGGraph":
        """Build or reload a single topic-entity group's graph."""
        cfg = load_cfg or GraphLoadConfig()
        instance = cls(topic_entity, dataset_name, split, global_config)

        if os.path.exists(instance._graphml_xml_file) and instance._graph.number_of_nodes() > 0:
            logger.info(
                f"[Cache HIT ] '{topic_entity}' — "
                f"{instance._graph.number_of_nodes()}N / "
                f"{instance._graph.number_of_edges()}E "
                f"← {instance._graphml_xml_file}"
            )
        else:
            logger.info(f"[Cache MISS] '{topic_entity}' — building from {len(q_ids)} questions …")
            instance._graph = _build_group_graph(q_ids, raw, cfg, _safe_name(topic_entity))

            if instance._graph.number_of_nodes() == 0:
                logger.error(
                    f"  Graph for '{topic_entity}' is EMPTY after loading. "
                    f"  Affected q_ids (first 5): {q_ids[:5]}"
                )
            else:
                # Persist immediately so subsequent runs hit the cache
                logger.info(f"  Caching graph → {instance._graphml_xml_file}")
                NetworkXStorage.write_nx_graph(instance._graph, instance._graphml_xml_file)

        return instance

    # ------------------------------------------------------------------ #
    # Factory: all groups                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_all_groups(
        cls,
        retrieval_file: str,
        dataset_name: str,
        split: str,
        global_config: dict,
        load_cfg: Optional[GraphLoadConfig] = None,
        strategy: str = PRIMARY_ENTITY_STRATEGY,
        topic_entity_filter: Optional[List[str]] = None,
    ) -> Dict[str, "SubgraphRAGGraph"]:
        """
        Load all topic-entity groups from a .pth retrieval file.

        Returns
        -------
        { topic_entity: SubgraphRAGGraph }

        Parameters
        ----------
        topic_entity_filter : only load these entities (useful for resuming).
        """
        cfg = load_cfg or GraphLoadConfig()
        global_config = _patch_global_config(global_config, dataset_name, split)

        # 1. Load raw .pth
        pth_path = _resolve_pth(cfg.retrieval_dir, retrieval_file)
        raw = _load_pth(pth_path)

        # 2. Group
        groups = _group_by_q_entity(raw, strategy)

        if topic_entity_filter:
            before = len(groups)
            groups = {k: v for k, v in groups.items() if k in topic_entity_filter}
            logger.info(f"[Filter] {len(groups)} / {before} groups kept.")

        # 3. Build / cache each graph
        graphs: Dict[str, SubgraphRAGGraph] = {}
        total = len(groups)
        for idx, (entity, q_ids) in enumerate(groups.items(), 1):
            logger.info(f"── [{idx}/{total}] '{entity}' ({len(q_ids)} questions)")
            graphs[entity] = cls.load_group(
                topic_entity=entity,
                q_ids=q_ids,
                raw=raw,
                dataset_name=dataset_name,
                split=split,
                global_config=global_config,
                load_cfg=cfg,
            )

        # 4. Save q_id → topic_entity index for query-time lookup
        index_path = os.path.join(global_config["working_dir"], "graph_index.json")
        q_to_entity = {q_id: ent for ent, q_ids in groups.items() for q_id in q_ids}
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(q_to_entity, f, indent=2, ensure_ascii=False)
        logger.info(f"[Index] q_id→entity index saved → {index_path}")

        # Summary
        empty = [e for e, g in graphs.items() if g._graph.number_of_nodes() == 0]
        if empty:
            logger.warning(f"[Summary] {len(empty)} empty graphs: {empty[:10]}")
        logger.info(
            f"[Summary] {len(graphs)} graphs loaded | "
            f"{len(empty)} empty | "
            f"index at {index_path}"
        )
        return graphs

    # ------------------------------------------------------------------ #
    # Debug                                                               #
    # ------------------------------------------------------------------ #

    def print_summary(self):
        G = self._graph
        degrees = [d for _, d in G.degree()] if G.number_of_nodes() else []
        import statistics
        print(f"\n{'='*58}")
        print(f"  Topic entity : {self.topic_entity}")
        print(f"  Cache file   : {self._graphml_xml_file}")
        print(f"  Nodes        : {G.number_of_nodes()}")
        print(f"  Edges        : {G.number_of_edges()}")
        if degrees:
            print(f"  Degree       : min={min(degrees)}, max={max(degrees)}, "
                  f"mean={statistics.mean(degrees):.1f}")
        for nid, attrs in list(G.nodes(data=True))[:2]:
            print(f"  Node sample  : [{nid}] {dict(attrs)}")
        for u, v, attrs in list(G.edges(data=True))[:2]:
            print(f"  Edge sample  : {u} --[{attrs.get('description','?')[:40]}]--> {v} "
                  f"w={attrs.get('weight', 0):.3f}")
        print(f"{'='*58}\n")

if __name__ == "__main__":
    # Quick smoke test for the first 3 groups
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test SubgraphRAGGraph loading.")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp", help="Dataset name (e.g., webqsp, cwq)")
    parser.add_argument("--split", type=str, default="test", help="Data split (e.g., test)")
    parser.add_argument("--retrieval_file", type=str, default="retrieval_result.pth", help="Name of the retrieval result file.")
    args = parser.parse_args()

    global_config = {"working_dir": "./hirag_cache_smoke_test"}
    graphs = SubgraphRAGGraph.load_all_groups(
        retrieval_file=args.retrieval_file,
        dataset_name=args.dataset_name,
        split=args.split,
        global_config=global_config,
        topic_entity_filter=None,  # set to e.g. ["Jamaica"] to test a single group
    )

    for entity, graph in list(graphs.items())[:3]:
        graph.print_summary()