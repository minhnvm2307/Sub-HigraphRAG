from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from graspologic.partition import hierarchical_leiden
import umap
from tqdm import tqdm

from .base import CommunityReport, EntityHit, QueryContext, QueryParam
from .prompt import PROMPTS
from .utils import clean_str, extract_first_complete_json, is_float_regex, list_of_list_to_csv, split_string_by_multi_markers

logger = logging.getLogger("retrieve.hirag")


def _full_entity_embeddings(sample: dict) -> list[tuple[str, torch.Tensor]]:
    text_entities = sample["text_entity_list"]
    entity_embs = sample["entity_embs"].float()
    pairs = []
    for entity_name, entity_emb in zip(text_entities, entity_embs):
        pairs.append((entity_name, entity_emb))
    return pairs


def ingest_sample(graph_storage, vector_index, sample: dict) -> None:
    entity_list = sample["text_entity_list"] + sample["non_text_entity_list"]
    relation_list = sample["relation_list"]
    sample_id = str(sample["id"])
    question = sample["question"]

    text_entity_embeddings = dict(_full_entity_embeddings(sample))
    logger.debug(
        "Ingest sample: id=%s question_len=%d text_entities=%d non_text_entities=%d triples=%d",
        sample_id,
        len(question),
        len(sample["text_entity_list"]),
        len(sample["non_text_entity_list"]),
        len(sample["h_id_list"]),
    )
    for entity_name in entity_list:
        node_type = "TEXT_ENTITY" if entity_name in text_entity_embeddings else "NON_TEXT_ENTITY"
        graph_storage.upsert_node(
            entity_name,
            id=entity_name,
            entity_name=entity_name,
            entity_type=node_type,
            description=entity_name,
            source_ids=json.dumps(sorted(set(json.loads(graph_storage.graph.nodes[entity_name]["source_ids"]) + [sample_id])))
            if graph_storage.graph.has_node(entity_name) and graph_storage.graph.nodes[entity_name].get("source_ids")
            else json.dumps([sample_id]),
        )

    edge_counts = Counter()
    for h_id, r_id, t_id in zip(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]):
        head = entity_list[h_id]
        relation = relation_list[r_id]
        tail = entity_list[t_id]
        edge_key = tuple(sorted((head, tail)))
        edge_counts[(edge_key[0], relation, edge_key[1])] += 1

    for (head, relation, tail), count in edge_counts.items():
        existing_samples = []
        if graph_storage.graph.has_edge(head, tail):
            existing = graph_storage.graph.edges[head, tail].get("source_ids")
            if existing:
                existing_samples = json.loads(existing)
        graph_storage.upsert_edge(
            head,
            tail,
            id=f"{head}|{relation}|{tail}",
            relation=relation,
            description=f"{head} --{relation}-- {tail}",
            weight=float(count),
            source_ids=json.dumps(sorted(set(existing_samples + [sample_id]))),
            question_example=question,
        )

    vector_index.upsert(
        (entity_name, emb, {"entity_name": entity_name, "sample_id": sample_id})
        for entity_name, emb in text_entity_embeddings.items()
    )
    logger.debug(
        "Sample ingested: id=%s graph_nodes=%d graph_edges=%d",
        sample_id,
        graph_storage.graph.number_of_nodes(),
        graph_storage.graph.number_of_edges(),
    )


def _summarize_members(graph_storage, members: list[str], max_members: int = 8) -> str:
    degrees = sorted(((node, graph_storage.graph.degree(node)) for node in members), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in degrees[:max_members]]
    return ", ".join(top_nodes)


def _global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: int = 15, metric: str = "cosine") -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    n_components = max(2, min(dim, len(embeddings) - 2))
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric).fit_transform(embeddings)


def _fit_gaussian_mixture(n_components: int, embeddings: np.ndarray, random_state: int) -> float:
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        init_params="k-means++",
    )
    gm.fit(embeddings)
    return gm.bic(embeddings)


def _get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 0, rel_tol: float = 1e-3) -> int:
    max_clusters = min(len(embeddings), max_clusters)
    if max_clusters <= 1:
        return 1
    n_clusters = np.arange(1, max_clusters)
    bics = []
    prev_bic = None
    for n in tqdm(n_clusters):
        bic = _fit_gaussian_mixture(n, embeddings, random_state)
        bics.append(bic)
        # early stop
        if prev_bic is not None and np.isfinite(prev_bic) and np.isfinite(bic):
            denom = max(abs(prev_bic), np.finfo(float).eps)
            if (abs(prev_bic - bic) / denom) < rel_tol:
                break
        prev_bic = bic
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def _gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0) -> tuple[list[np.ndarray], int]:
    n_clusters = _get_optimal_clusters(embeddings, random_state=random_state)
    gm = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        n_init=5,
        init_params="k-means++",
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def _perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> list[np.ndarray]:
    reduced = _global_cluster_embeddings(embeddings, dim=min(dim, max(2, len(embeddings) - 2)))
    clusters, n_global_clusters = _gmm_cluster(reduced, threshold)

    logger.info(f"Global Clusters (GMM): {n_global_clusters}")

    all_clusters = [[] for _ in range(len(embeddings))]
    for cluster_id in range(n_global_clusters):
        for idx, labels in enumerate(clusters):
            if cluster_id in labels:
                all_clusters[idx].append(cluster_id)
    return [np.array(cluster) for cluster in all_clusters]


def _handle_single_entity_extraction(record_attributes: list[str], source_id: str) -> dict | None:
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    return {
        "entity_name": entity_name,
        "entity_type": clean_str(record_attributes[2].upper()),
        "description": clean_str(record_attributes[3]),
        "source_id": source_id,
    }


def _handle_single_relationship_extraction(record_attributes: list[str], source_id: str) -> dict | None:
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    return {
        "src_id": clean_str(record_attributes[1].upper()),
        "tgt_id": clean_str(record_attributes[2].upper()),
        "description": clean_str(record_attributes[3]),
        "weight": float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
        "source_id": source_id,
    }


def _cluster_source_ids(graph_storage, members: list[str]) -> list[str]:
    source_ids = []
    for member in members:
        node = graph_storage.graph.nodes.get(member, {})
        raw = node.get("source_ids")
        if raw:
            source_ids.extend(json.loads(raw))
    return sorted(set(source_ids))


def _truncate_text(value: object, max_chars: int) -> str:
    text = clean_str(str(value or ""))
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _fit_rows_to_char_budget(rows: list[list], max_chars: int) -> list[list]:
    if not rows:
        return rows
    if max_chars <= 0:
        return rows[:1]

    selected = [rows[0]]
    current_chars = len(list_of_list_to_csv(selected))
    for row in rows[1:]:
        row_text = list_of_list_to_csv([row])
        if current_chars + 1 + len(row_text) > max_chars:
            break
        selected.append(row)
        current_chars += 1 + len(row_text)
    return selected


def _truncate_cluster_nodes(cluster_nodes: list[dict], max_length_in_cluster: int) -> list[dict]:
    selected = cluster_nodes[:]
    while selected:
        total_length = sum(len(node["entity_name"]) + len(node["description"]) for node in selected)
        if total_length <= max_length_in_cluster:
            break
        next_size = max(1, int(len(selected) * 0.8))
        selected = random.sample(selected, next_size)
    return selected


def _build_cluster_prompt(cluster_nodes: list[dict], meta_entity_types: list[str]) -> str:
    context = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "meta_attribute_list": meta_entity_types,
        "entity_description_list": ",".join(
            f"({node['entity_name']}, {node['description']})" for node in cluster_nodes
        ),
    }
    return PROMPTS["summary_clusters"].format(**context)


def _parse_summary_output(output: str, source_ids: list[str]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    records = split_string_by_multi_markers(
        output,
        [PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"]],
    )
    maybe_nodes: dict[str, dict] = {}
    maybe_edges: dict[tuple[str, str], dict] = {}
    merged_source_id = "<SEP>".join(source_ids)
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is None:
            continue
        record_attributes = split_string_by_multi_markers(match.group(1), [PROMPTS["DEFAULT_TUPLE_DELIMITER"]])
        entity_row = _handle_single_entity_extraction(record_attributes, merged_source_id)
        if entity_row is not None:
            maybe_nodes[entity_row["entity_name"]] = entity_row
            continue
        relation_row = _handle_single_relationship_extraction(record_attributes, merged_source_id)
        if relation_row is not None:
            maybe_edges[(relation_row["src_id"], relation_row["tgt_id"])] = relation_row
    return maybe_nodes, maybe_edges


def _fallback_cluster_output(cluster_nodes: list[dict], source_ids: list[str]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    source_id = "<SEP>".join(source_ids)
    core_name = ", ".join(node["entity_name"] for node in cluster_nodes[:3])
    meta_name = f"CLUSTER_{abs(hash('|'.join(sorted(node['entity_name'] for node in cluster_nodes)))) % 10**8}"
    nodes = {
        meta_name: {
            "entity_name": meta_name,
            "entity_type": "ORGANIZATION",
            "description": f"Summarized semantic group related to: {core_name}",
            "source_id": source_id,
        }
    }
    edges = {}
    for node in cluster_nodes:
        edges[(node["entity_name"], meta_name)] = {
            "src_id": node["entity_name"],
            "tgt_id": meta_name,
            "description": f"{node['entity_name']} belongs to semantic group {meta_name}",
            "weight": 1.0,
            "source_id": source_id,
        }
    return nodes, edges


def semantic_cluster_layers(
    graph_storage,
    vector_index,
    embedding_func,
    llm_callable=None,
    max_layers: int = 3,
    min_cluster_size: int = 3,
    reduction_dimension: int = 2,
    cluster_threshold: float = 0.1,
    stop_sparsity_threshold: float = 0.98,
    stop_sparsity_change_rate: float = 0.05,
    max_length_in_cluster: int = 60000,
    meta_entity_types: list[str] | None = None,
) -> list[dict]:
    meta_entity_types = meta_entity_types or PROMPTS["META_ENTITY_TYPES"]
    if vector_index.embeddings.numel() == 0 or len(vector_index.ids) < min_cluster_size:
        logger.info(
            "Skip semantic clustering: entities=%d min_cluster_size=%d",
            len(vector_index.ids),
            min_cluster_size,
        )
        return []

    nodes = []
    for node_id in vector_index.ids:
        metadata = vector_index.meta.get(node_id, {})
        node_data = graph_storage.graph.nodes.get(node_id)
        if node_data is None:
            continue
        nodes.append(
            {
                "entity_name": node_id,
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_id": "<SEP>".join(json.loads(node_data.get("source_ids", "[]"))),
                "embedding": vector_index.embeddings[vector_index.ids.index(node_id)].numpy(),
                "metadata": metadata,
            }
        )

    hierarchical_layers = []
    pre_cluster_sparsity = 0.01
    logger.info("Start method-1 semantic clustering: entities=%d max_layers=%d", len(nodes), max_layers)

    for layer_idx in range(max_layers):
        if len(nodes) < min_cluster_size:
            logger.info("Stop semantic clustering at layer=%d because nodes=%d", layer_idx, len(nodes))
            break
        embeddings = np.array([node["embedding"] for node in nodes])
        clusters = _perform_clustering(embeddings, dim=reduction_dimension, threshold=cluster_threshold)
        flattened = np.concatenate(clusters) if clusters else np.array([])
        if len(flattened) == 0:
            logger.info("Stop semantic clustering at layer=%d because no clusters were assigned", layer_idx)
            break
        unique_clusters = np.unique(flattened)
        cluster_sizes = Counter(flattened)
        cluster_sparsity = 1 - sum(size * (size - 1) for size in cluster_sizes.values()) / max(len(nodes) * max(len(nodes) - 1, 1), 1)
        cluster_sparsity_change_rate = abs(cluster_sparsity - pre_cluster_sparsity) / (pre_cluster_sparsity + 1e-8)
        pre_cluster_sparsity = cluster_sparsity
        logger.info(
            "Semantic clustering layer=%d labels=%d nodes=%d sparsity=%.4f delta=%.4f",
            layer_idx,
            len(unique_clusters),
            len(nodes),
            cluster_sparsity,
            cluster_sparsity_change_rate,
        )
        if cluster_sparsity >= stop_sparsity_threshold:
            logger.info("Stop semantic clustering at layer=%d due to sparsity threshold", layer_idx)
            break
        if cluster_sparsity_change_rate <= stop_sparsity_change_rate:
            logger.info("Stop semantic clustering at layer=%d due to sparsity change threshold", layer_idx)
            break

        next_layer_nodes = []
        layer_clusters = []
        for label in unique_clusters:
            indices = [idx for idx, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[idx] for idx in indices]
            logger.info("Semantic cluster layer=%d label=%s size=%d", layer_idx, int(label), len(cluster_nodes))
            if len(cluster_nodes) < min_cluster_size:
                continue

            cluster_nodes = _truncate_cluster_nodes(cluster_nodes, max_length_in_cluster=max_length_in_cluster)
            source_ids = sorted(set(source_id for node in cluster_nodes for source_id in node["source_id"].split("<SEP>") if source_id))
            prompt = _build_cluster_prompt(cluster_nodes, meta_entity_types)

            if llm_callable is None:
                logger.warning("No LLM configured for semantic summary; using fallback output for layer=%d label=%s", layer_idx, int(label))
                parsed_nodes, parsed_edges = _fallback_cluster_output(cluster_nodes, source_ids)
            else:
                llm_output = llm_callable(prompt)
                parsed_nodes, parsed_edges = _parse_summary_output(llm_output, source_ids)
                if not parsed_nodes:
                    logger.warning("LLM semantic summary produced no entities; using fallback output for layer=%d label=%s", layer_idx, int(label))
                    parsed_nodes, parsed_edges = _fallback_cluster_output(cluster_nodes, source_ids)

            new_entities = list(parsed_nodes.values())
            logger.info(
                "Semantic cluster layer=%d label=%s parsed_entities=%d parsed_relations=%d",
                layer_idx,
                int(label),
                len(parsed_nodes),
                len(parsed_edges),
            )
            if new_entities:
                new_embeddings = embedding_func([entity["description"] for entity in new_entities]).float()
            else:
                new_embeddings = torch.empty(0)

            for entity, embedding in zip(new_entities, new_embeddings):
                entity["embedding"] = embedding
                node_id = entity["entity_name"]
                graph_storage.upsert_node(
                    node_id,
                    id=node_id,
                    entity_name=node_id,
                    entity_type=entity["entity_type"],
                    description=entity["description"],
                    source_ids=json.dumps(source_ids),
                    semantic_layer=int(layer_idx),
                    members=json.dumps(sorted(node["entity_name"] for node in cluster_nodes)),
                )
                next_layer_nodes.append(entity)

            for relation in parsed_edges.values():
                src_id = relation["src_id"]
                tgt_id = relation["tgt_id"]
                if not graph_storage.graph.has_node(src_id):
                    continue
                if not graph_storage.graph.has_node(tgt_id):
                    continue
                graph_storage.upsert_edge(
                    src_id,
                    tgt_id,
                    id=f"{src_id}|semantic_relation|{tgt_id}",
                    relation="semantic_relation",
                    description=relation["description"],
                    weight=float(relation["weight"]),
                    source_ids=json.dumps(source_ids),
                    semantic_layer=int(layer_idx),
                )

            vector_index.upsert(
                (entity["entity_name"], entity["embedding"], {"entity_name": entity["entity_name"], "semantic_layer": layer_idx})
                for entity in new_entities
            )
            layer_clusters.append(
                {
                    "label": int(label),
                    "members": sorted(node["entity_name"] for node in cluster_nodes),
                    "generated_entities": sorted(entity["entity_name"] for entity in new_entities),
                    "generated_relations": len(parsed_edges),
                }
            )

        if not next_layer_nodes:
            logger.info("Semantic clustering produced no next-layer entities at layer=%d", layer_idx)
            break

        hierarchical_layers.append({"layer": layer_idx, "clusters": layer_clusters})
        dedup = {}
        for node in next_layer_nodes:
            dedup[node["entity_name"]] = node
        nodes = list(dedup.values())
        logger.info("Semantic clustering layer=%d completed: next_entities=%d", layer_idx, len(nodes))
        if len(nodes) <= 2:
            logger.info("Stop semantic clustering at layer=%d because deduplicated entities=%d", layer_idx, len(nodes))
            break

    logger.info("Semantic clustering finished: layers=%d", len(hierarchical_layers))
    return hierarchical_layers


def structural_communities(graph_storage) -> dict[str, dict]:
    graph = graph_storage.graph
    if graph.number_of_nodes() == 0:
        logger.info("Skip structural community detection: graph is empty")
        return {}

    logger.info("Run structural community detection with hierarchical_leiden")
    community_rows = hierarchical_leiden(
        nx.Graph(graph),
        max_cluster_size=max(10, graph.number_of_nodes() // 5)
    )
    node_clusters = defaultdict(list)
    for row in community_rows:
        node_clusters[row.node].append({"level": int(row.level), "cluster": str(row.cluster)})

    community_map = defaultdict(lambda: {"level": 0, "nodes": set(), "edges": set()})
    for node_id, clusters in node_clusters.items():
        graph.nodes[node_id]["clusters"] = json.dumps(clusters)
        for cluster in clusters:
            key = cluster["cluster"]
            community_map[key]["level"] = int(cluster["level"])
            community_map[key]["nodes"].add(node_id)

    for community_id, payload in community_map.items():
        for node_id in payload["nodes"]:
            for neighbor in graph.neighbors(node_id):
                if neighbor in payload["nodes"]:
                    edge = tuple(sorted((node_id, neighbor)))
                    payload["edges"].add(edge)

    result = {
        community_id: {
            "community_id": community_id,
            "level": payload["level"],
            "nodes": sorted(payload["nodes"]),
            "edges": [f"{src}|{tgt}" for src, tgt in sorted(payload["edges"])],
        }
        for community_id, payload in community_map.items()
    }
    logger.info(
        "Structural community detection (Leiden) finished: No. of communities=%d nodes=%d",
        len(result),
        len(node_clusters),
    )
    return result


def build_community_reports(
    graph_storage,
    communities: dict[str, dict],
    llm_callable=None,
    max_input_chars: int = 9000,
) -> dict[str, dict]:
    reports = {}
    logger.info("Build community reports: communities=%d", len(communities))
    for community_id, payload in communities.items():
        node_names = payload["nodes"]

        # ── collect edge labels ──────────────────────────────────────────────
        edge_labels = []
        valid_edges = []  # only edges we can actually look up
        for edge_id in payload["edges"]:
            parts = edge_id.rsplit("|", 1)
            if len(parts) != 2:
                logger.warning(
                    "Malformed edge_id (cannot split on '|'): %r in community=%s",
                    edge_id, community_id,
                )
                continue
            src, tgt = parts
            # NetworkX undirected graph may store (tgt, src) instead of (src, tgt)
            if graph_storage.graph.has_edge(src, tgt):
                data = graph_storage.graph.edges[src, tgt]
                valid_edges.append((src, tgt, edge_id))
            elif graph_storage.graph.has_edge(tgt, src):
                data = graph_storage.graph.edges[tgt, src]
                valid_edges.append((tgt, src, edge_id))
            else:
                logger.warning(
                    "Edge not found in graph: src=%r tgt=%r edge_id=%r community=%s",
                    src, tgt, edge_id, community_id,
                )
                continue
            edge_labels.append(data.get("relation", "related_to"))

        relation_counts = Counter(edge_labels)

        # ── entity rows ──────────────────────────────────────────────────────
        rows = [["id", "entity", "type", "description"]]
        for idx, node_name in enumerate(node_names):
            node_data = graph_storage.graph.nodes[node_name]
            rows.append(
                [
                    idx,
                    node_name,
                    node_data.get("entity_type", "UNKNOWN"),
                    _truncate_text(node_data.get("description", ""), 160),
                ]
            )

        # ── relation rows (use only validated edges) ─────────────────────────
        relation_rows = [["id", "source", "target", "relation", "description"]]
        for idx, (src, tgt, _edge_id) in enumerate(valid_edges):
            edge_data = graph_storage.graph.edges[src, tgt]
            relation_rows.append(
                [
                    idx,
                    src,
                    tgt,
                    edge_data.get("relation", "related_to"),
                    _truncate_text(edge_data.get("description", ""), 120),
                ]
            )

        # ── truncate to char budget ──────────────────────────────────────────
        original_entity_count = len(rows) - 1
        original_relation_count = len(relation_rows) - 1
        entity_budget = max(1000, int(max_input_chars * 0.4))
        rows = _fit_rows_to_char_budget(rows, max_chars=entity_budget)
        remaining_budget = max_input_chars - len(list_of_list_to_csv(rows))
        relation_rows = _fit_rows_to_char_budget(relation_rows, max_chars=remaining_budget)
        community_input = (
            f"Entities:\n```csv\n{list_of_list_to_csv(rows)}\n```\n"
            f"Relationships:\n```csv\n{list_of_list_to_csv(relation_rows)}\n```"
        )
        if len(rows) - 1 != original_entity_count or len(relation_rows) - 1 != original_relation_count:
            logger.warning(
                "Truncated community report input: community=%s entities=%d/%d relations=%d/%d chars=%d",
                community_id,
                len(rows) - 1,
                original_entity_count,
                len(relation_rows) - 1,
                original_relation_count,
                len(community_input),
            )

        # ── LLM report ───────────────────────────────────────────────────────
        report_json = None
        if llm_callable is not None:
            try:
                llm_output = llm_callable(PROMPTS["community_report_paper"].format(input_text=community_input))
                report_json = extract_first_complete_json(llm_output)
            except Exception as exc:
                logger.warning("Community report LLM failed: community=%s error=%s", community_id, exc)

        if report_json is None:
            summary = (
                f"Community {community_id} contains {len(node_names)} nodes and "
                f"{len(valid_edges)} edges. Top entities: {_summarize_members(graph_storage, node_names)}. "
                f"Top relations: {', '.join(rel for rel, _ in relation_counts.most_common(5))}."
            )
            report_json = {
                "title": f"Community {community_id}",
                "summary": summary,
                "rating": float(min(10.0, len(valid_edges))),
                "rating_explanation": "Fallback report generated without LLM.",
                "findings": [],
            }

        reports[community_id] = {
            "community_id": community_id,
            "level": payload["level"],
            "title": report_json.get("title", f"Community {community_id}"),
            "summary": report_json.get("summary", ""),
            "node_ids": node_names,
            "edge_ids": payload["edges"],
            "rating": float(report_json.get("rating", len(valid_edges))),
            "metadata": {
                "relation_counts": dict(relation_counts),
                "rating_explanation": report_json.get("rating_explanation", ""),
                "findings": report_json.get("findings", []),
            },
        }
    logger.info("Community reports built: reports=%d", len(reports))
    return reports


def _entity_hits(graph_storage, vector_index, query_embedding: torch.Tensor, top_k: int) -> list[EntityHit]:
    hits = []
    for row in vector_index.query(query_embedding, top_k=top_k):
        node_id = row["id"]
        hits.append(
            EntityHit(
                node_id=node_id,
                entity_name=row["metadata"].get("entity_name", node_id),
                score=row["score"],
                node_data=graph_storage.node_data(node_id),
            )
        )
    logger.debug("#Step 1: Entity query finished: top_k=%d hits=%d", top_k, len(hits))
    return hits


def _local_context(graph_storage, hits: list[EntityHit], top_k_triples: int) -> list[dict]:
    triples = []
    seen = set()
    for hit in hits:
        for row in graph_storage.neighbors_with_edges(hit.node_id):
            edge_id = row.get("id")
            if edge_id in seen:
                continue
            seen.add(edge_id)
            triples.append(
                {
                    "source": row["source"],
                    "relation": row.get("relation", "related_to"),
                    "target": row["target"],
                    "weight": float(row.get("weight", 0.0)),
                }
            )
    triples.sort(key=lambda x: x["weight"], reverse=True)
    result = triples[:top_k_triples]
    logger.debug("Local context built: seed_hits=%d triples=%d", len(hits), len(result))
    return result


def _global_context(hits: list[EntityHit], community_store, community_level: int, top_k_communities: int) -> list[CommunityReport]:
    counts = Counter()
    for hit in hits:
        for cluster in json.loads(hit.node_data.get("clusters", "[]")):
            if int(cluster["level"]) <= community_level:
                counts[str(cluster["cluster"])] += 1
    reports = []
    for community_id, count in counts.most_common(top_k_communities):
        row = community_store.get(community_id)
        if not row:
            continue
        reports.append(
            CommunityReport(
                community_id=row["community_id"],
                level=int(row["level"]),
                title=row["title"],
                summary=row["summary"],
                node_ids=row["node_ids"],
                edge_ids=row["edge_ids"],
                rating=float(row["rating"]) + float(count),
                metadata=row.get("metadata", {}),
            )
        )
    logger.debug(
        "Global context built: seed_hits=%d matched_communities=%d returned=%d",
        len(hits),
        len(counts),
        len(reports),
    )
    return reports


def _bridge_context(graph_storage, hits: list[EntityHit], max_paths: int, max_hops: int) -> list[dict]:
    seed_ids = [hit.node_id for hit in hits]
    bridges = []
    seen = set()
    for idx, source in enumerate(seed_ids):
        for target in seed_ids[idx + 1 :]:
            for path in graph_storage.shortest_paths(source, target, cutoff=max_hops):
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                bridges.append({"path": path, "hops": len(path) - 1})
                if len(bridges) >= max_paths:
                    logger.debug(
                        "Bridge context reached limit: seed_hits=%d paths=%d",
                        len(hits),
                        len(bridges),
                    )
                    return bridges
    logger.debug("Bridge context built: seed_hits=%d paths=%d", len(hits), len(bridges))
    return bridges


def _safe_json_loads(value: str) -> list:
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def _node_rank(graph_storage, node_id: str) -> int:
    if not graph_storage.graph.has_node(node_id):
        return 0
    return int(graph_storage.graph.degree(node_id))


def _edge_rank(graph_storage, src: str, tgt: str) -> int:
    return _node_rank(graph_storage, src) + _node_rank(graph_storage, tgt)


def _query_similarity_map(vector_index, query_embedding: torch.Tensor) -> dict[str, float]:
    logger.debug("#Step 3: Calculating query similarity map: query_dim=%d index_size=%d", query_embedding.shape[0], len(vector_index.ids))
    if vector_index.embeddings.numel() == 0 or not vector_index.ids:
        return {}
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]
    query_norm = torch.nn.functional.normalize(query_embedding.reshape(1, -1).float(), dim=1)
    index_norm = torch.nn.functional.normalize(vector_index.embeddings.float(), dim=1)
    scores = torch.mm(index_norm, query_norm.t()).squeeze(1).tolist()
    return {node_id: float(score) for node_id, score in zip(vector_index.ids, scores)}


def _collect_source_ids(node_datas: list[dict]) -> list[str]:
    sources = []
    for node in node_datas:
        sources.extend(_safe_json_loads(node.get("source_ids", "")))
    return sorted(set(str(source) for source in sources if str(source)))


def _summarize_communities(hits: list[EntityHit], community_store, level: int, top_k: int) -> list[dict]:
    logger.debug("#Step 2: Summarizing communities: seed_hits=%d community_level=%d", len(hits), level)
    counts = Counter()
    for hit in hits:
        for cluster in _safe_json_loads(hit.node_data.get("clusters", "[]")):
            if int(cluster.get("level", 0)) <= level:
                counts[str(cluster.get("cluster"))] += 1

    communities = []
    for community_id, count in counts.most_common(top_k):
        row = community_store.get(community_id)
        if not row:
            continue
        summary = row.get("summary", "")
        title = row.get("title", f"Community {community_id}")
        communities.append(
            {
                "community_id": row.get("community_id", community_id),
                "level": int(row.get("level", 0)),
                "title": title,
                "summary": summary,
                "rating": float(row.get("rating", 0.0)) + float(count),
                "report_string": f"{title}: {summary}".strip(),
                "nodes": row.get("node_ids", []),
            }
        )
    return communities


def _community_member_boost(communities: list[dict]) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for community in communities:
        rating = float(community.get("rating", 0.0))
        for node_id in community.get("nodes", []):
            boosts[node_id] = max(boosts.get(node_id, 0.0), rating)
    return boosts

def _entity_scores_from_scored_triples(sample_id: str, mlp_results: dict[str, list[dict]], top_k: int) -> dict[str, float]:
    if sample_id not in mlp_results:
        return {}
    h_list, r_list, t_list, score_list = zip(*mlp_results[sample_id]['scored_triples'])
    entity_scores = defaultdict(float)
    for h, r, t, s in zip(h_list, r_list, t_list, score_list):
        entity_scores[h] = max(entity_scores[h], s)
        entity_scores[t] = max(entity_scores[t], s)
    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_entities[:top_k])

def _score_node(
    node_id: str,
    query_sim: dict[str, float],
    graph_storage,
    community_boost: dict[str, float],
    use_query_scores: bool,
) -> float:
    sim = query_sim.get(node_id, 0.0) if use_query_scores else 0.0
    degree = _node_rank(graph_storage, node_id)
    boost = community_boost.get(node_id, 0.0)
    return sim + 0.02 * degree + 0.05 * boost


def _score_edge(
    src: str,
    tgt: str,
    edge_data: dict,
    query_sim: dict[str, float],
    graph_storage,
    community_boost: dict[str, float],
    use_query_scores: bool,
) -> float:
    sim = 0.5 * (query_sim.get(src, 0.0) + query_sim.get(tgt, 0.0)) if use_query_scores else 0.0
    weight = float(edge_data.get("weight", 0.0))
    degree = _edge_rank(graph_storage, src, tgt)
    boost = 0.5 * (community_boost.get(src, 0.0) + community_boost.get(tgt, 0.0))
    return sim + 0.03 * weight + 0.01 * degree + 0.03 * boost


def _select_key_entities(
    hits: list[EntityHit],
    query_sim: dict[str, float],
    graph_storage,
    community_boost: dict[str, float],
    top_m: int,
    use_query_scores: bool,
) -> list[str]:
    scored = []
    for hit in hits:
        score = _score_node(
            hit.node_id,
            query_sim,
            graph_storage,
            community_boost,
            use_query_scores=use_query_scores,
        )
        scored.append((score, hit.node_id))
    scored.sort(reverse=True)
    return [node_id for _, node_id in scored[: max(1, top_m)]]


def _path_score(
    graph_storage,
    path: list[str],
    query_sim: dict[str, float],
    community_boost: dict[str, float],
    use_query_scores: bool,
) -> float:
    if len(path) < 2:
        return 0.0
    edge_scores = []
    for src, tgt in zip(path, path[1:]):
        if not graph_storage.graph.has_edge(src, tgt):
            continue
        edge_data = graph_storage.edge_data(src, tgt)
        edge_scores.append(
            _score_edge(
                src,
                tgt,
                edge_data,
                query_sim,
                graph_storage,
                community_boost,
                use_query_scores=use_query_scores,
            )
        )
    if not edge_scores:
        return 0.0
    coherence = sum(edge_scores) / len(edge_scores)
    unique_nodes = len(set(path))
    degree_penalty = 0.02 * sum(_node_rank(graph_storage, node_id) for node_id in path)
    diversity_bonus = 0.05 * unique_nodes
    return sum(edge_scores) + 0.5 * coherence + diversity_bonus - degree_penalty


def _guided_paths(
    graph_storage,
    sources: list[str],
    targets: list[str],
    node_scores: dict[str, float],
    max_paths: int,
    max_hops: int,
    query_sim: dict[str, float],
    community_boost: dict[str, float],
    use_query_scores: bool,
) -> list[dict]:
    if not sources or not targets:
        return []
    paths = []
    seen = set()
    for source in sources:
        for target in targets:
            if source == target:
                continue
            if not graph_storage.graph.has_node(source) or not graph_storage.graph.has_node(target):
                continue
            # Best-first search guided by path scores.
            frontier = [(0.0, [source])]
            while frontier and len(paths) < max_paths:
                frontier.sort()
                neg_score, path = frontier.pop(0)
                last = path[-1]
                if last == target:
                    key = tuple(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append({"path": path, "hops": len(path) - 1})
                    continue
                if len(path) - 1 >= max_hops:
                    continue
                for neighbor in graph_storage.graph.neighbors(last):
                    if neighbor in path:
                        continue
                    if neighbor not in node_scores:
                        node_scores[neighbor] = _score_node(
                            neighbor,
                            query_sim,
                            graph_storage,
                            community_boost,
                            use_query_scores=use_query_scores,
                        )
                    candidate = path + [neighbor]
                    score = _path_score(
                        graph_storage,
                        candidate,
                        query_sim,
                        community_boost,
                        use_query_scores=use_query_scores,
                    )
                    frontier.append((-score, candidate))
            if len(paths) >= max_paths:
                return paths
    return paths


def _collect_entity_rows(graph_storage, hits: list[EntityHit]) -> list[dict]:
    rows = []
    for hit in hits:
        data = hit.node_data or {}
        rows.append(
            {
                "entity_name": hit.entity_name,
                "entity_type": data.get("entity_type", "UNKNOWN"),
                "description": data.get("description", "UNKNOWN"),
                "rank": _node_rank(graph_storage, hit.node_id),
                "source_ids": data.get("source_ids", ""),
            }
        )
    return rows


def _collect_relation_rows(graph_storage, hits: list[EntityHit], top_k: int) -> list[dict]:
    edges = []
    seen = set()
    for hit in hits:
        for row in graph_storage.neighbors_with_edges(hit.node_id):
            src = row.get("source")
            tgt = row.get("target")
            key = tuple(sorted((src, tgt)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "src_tgt": (src, tgt),
                    "description": row.get("description", "related_to"),
                    "weight": float(row.get("weight", 0.0)),
                    "rank": _edge_rank(graph_storage, src, tgt),
                }
            )
    edges.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)
    return edges[:top_k]


def _collect_unified_edges(
    graph_storage,
    hits: list[EntityHit],
    bridge_paths: list[dict],
    query_sim: dict[str, float],
    community_boost: dict[str, float],
    top_k: int,
    min_score: float = 0.0,
    use_query_scores: bool = True,
) -> list[dict]:
    edges = []
    # Local edges.
    for hit in hits:
        for row in graph_storage.neighbors_with_edges(hit.node_id):
            src = row.get("source")
            tgt = row.get("target")
            if not src or not tgt:
                continue
            score = _score_edge(
                src,
                tgt,
                row,
                query_sim,
                graph_storage,
                community_boost,
                use_query_scores=use_query_scores,
            )
            if score < min_score:
                continue
            edges.append(
                {
                    "src_tgt": (src, tgt),
                    "relation": row.get("relation", "related_to"),
                    "description": row.get("description", "related_to"),
                    "weight": float(row.get("weight", 0.0)),
                    "rank": score,
                }
            )

    # Bridge edges from guided paths.
    for path_row in bridge_paths:
        path = path_row.get("path", [])
        for src, tgt in zip(path, path[1:]):
            if not graph_storage.graph.has_edge(src, tgt):
                continue
            data = graph_storage.edge_data(src, tgt)
            score = _score_edge(
                src,
                tgt,
                data,
                query_sim,
                graph_storage,
                community_boost,
                use_query_scores=use_query_scores,
            )
            if score < min_score:
                continue
            edges.append(
                {
                    "src_tgt": (src, tgt),
                    "relation": data.get("relation", "related_to"),
                    "description": data.get("description", "related_to"),
                    "weight": float(data.get("weight", 0.0)),
                    "rank": score,
                }
            )

    # Deduplicate only if every attribute matches.
    dedup = {}
    for edge in edges:
        key = (
            edge["src_tgt"][0],
            edge["src_tgt"][1],
            edge.get("relation", ""),
            edge.get("description", ""),
        )
        existing = dedup.get(key)
        if existing is None or edge["rank"] > existing["rank"]:
            dedup[key] = edge

    ranked = list(dedup.values())
    ranked.sort(key=lambda x: x["rank"], reverse=True)
    return ranked[:top_k]


def _collect_reasoning_path_edges(
    graph_storage,
    paths: list[dict],
    top_k: int,
    query_sim: dict[str, float],
    community_boost: dict[str, float],
    use_query_scores: bool,
) -> list[dict]:
    edges = []
    seen = set()
    for path_row in paths:
        path = path_row.get("path", [])
        for src, tgt in zip(path, path[1:]):
            key = tuple(sorted((src, tgt)))
            if key in seen:
                continue
            seen.add(key)
            data = graph_storage.edge_data(src, tgt) if graph_storage.graph.has_edge(src, tgt) else {}
            rank = _score_edge(
                src,
                tgt,
                data,
                query_sim,
                graph_storage,
                community_boost,
                use_query_scores=use_query_scores,
            )
            edges.append(
                {
                    "src_tgt": (src, tgt),
                    "relation": data.get("relation", "related_to"),
                    "description": data.get("description", "related_to"),
                    "weight": float(data.get("weight", 0.0)),
                    "rank": rank,
                }
            )
    edges.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)
    return edges[:top_k]


def _render_context_sections(
    communities: list[dict],
    entities: list[dict],
    relations: list[dict],
    reasoning_path: list[dict],
    sources: list[str],
    mode: str,
) -> str:
    sections = []
    if mode in {"hi", "hi_global"}:
        community_rows = [["id", "content"]]
        for idx, c in enumerate(communities):
            community_rows.append([idx, c.get("report_string", "")])
        sections.append(("Backgrounds", list_of_list_to_csv(community_rows)))
    if mode in {"hi", "hi_bridge"}:
        path_rows = [["id", "source", "target", "relation", "description", "weight", "rank"]]
        for idx, e in enumerate(reasoning_path):
            path_rows.append(
                [
                    idx,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e.get("relation", "related_to"),
                    e.get("description", ""),
                    e.get("weight", 0.0),
                    e.get("rank", 0),
                ]
            )
        sections.append(("Reasoning Path", list_of_list_to_csv(path_rows)))
    if mode in {"hi", "hi_local", "hi_bridge"}:
        entity_rows = [["id", "entity", "type", "description", "rank"]]
        for idx, n in enumerate(entities):
            entity_rows.append(
                [
                    idx,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n.get("rank", 0),
                ]
            )
        sections.append(("Detail Entity Information", list_of_list_to_csv(entity_rows)))

        relation_rows = [["id", "source", "target", "relation", "description", "weight", "rank"]]
        for idx, e in enumerate(relations):
            relation_rows.append(
                [
                    idx,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e.get("relation", "related_to"),
                    e.get("description", ""),
                    e.get("weight", 0.0),
                    e.get("rank", 0),
                ]
            )
        sections.append(("Relations", list_of_list_to_csv(relation_rows)))

    source_rows = [["id", "content"]]
    for idx, source_id in enumerate(sources):
        source_rows.append([idx, source_id])
    sections.append(("Source Documents", list_of_list_to_csv(source_rows)))

    rendered = []
    for title, csv_payload in sections:
        rendered.append(f"-----{title}-----\n```csv\n{csv_payload}\n```")
    return "\n".join(rendered)


async def build_query_context(
    graph_storage,
    vector_index,
    community_store,
    question: str,
    query_embedding: torch.Tensor,
    param: QueryParam,
) -> QueryContext:
    top_k = param.top_k if param.top_k else param.top_k_entities
    top_k_relations = param.top_k_triples
    top_k_communities = param.top_k_communities
    level = param.level if param.level is not None else param.community_level
    logger.info(
        "Build query context: mode=%s top_k=%d top_k_relations=%d top_k_communities=%d",
        param.mode,
        top_k,
        top_k_relations,
        top_k_communities,
    )

    hits = _entity_hits(graph_storage, vector_index, query_embedding, top_k=top_k)
    communities = _summarize_communities(hits, community_store, level=level, top_k=top_k_communities)
    community_boost = _community_member_boost(communities)
    query_sim = _query_similarity_map(vector_index, query_embedding)

    entities = _collect_entity_rows(graph_storage, hits)

    # Optionally compute MLP-derived entity scores and merge with similarity
    mlp_entity_scores = {}
    if param.scoring_method in {"mlp", "hybrid"} and param.mlp_result_path:
        mlp_results = torch.load(param.mlp_result_path)
        if mlp_results:
            mlp_entity_scores = _entity_scores_from_scored_triples(
                param.sample_id, mlp_results, top_k=param.mlp_top_k
            )

    # Build initial similarity-based key entities
    key_entities = _select_key_entities(
        hits,
        query_sim=query_sim,
        graph_storage=graph_storage,
        community_boost=community_boost,
        top_m=param.top_m,
        use_query_scores=param.use_query_scores,
    )

    # Merge MLP entity scores into node scoring if requested
    if param.scoring_method in {"mlp", "hybrid"} and mlp_entity_scores:
        # Expand the candidate set with entities from mlp scoring
        merged_candidates = set(key_entities) | set(mlp_entity_scores.keys())
        scored = []
        for node_id in merged_candidates:
            sim_score = _score_node(node_id, query_sim, graph_storage, community_boost, use_query_scores=param.use_query_scores)
            mlp_score = mlp_entity_scores.get(node_id, 0.0)
            combined = sim_score + param.mlp_weight * mlp_score
            scored.append((combined, node_id))
        scored.sort(reverse=True)
        key_entities = [node_id for _, node_id in scored[: max(1, param.top_m)]]

    node_scores = {
        node_id: _score_node(
            node_id,
            query_sim,
            graph_storage,
            community_boost,
            use_query_scores=param.use_query_scores,
        ) + param.mlp_weight * mlp_entity_scores.get(node_id, 0.0)
        for node_id in key_entities
    }
    logger.debug("#Step 4: Guided bridge path retrieval: key_entities=%d", len(key_entities))
    bridge_paths = _guided_paths(
        graph_storage,
        sources=key_entities[: max(1, len(key_entities) // 2)],
        targets=key_entities[max(1, len(key_entities) // 2) :],
        node_scores=node_scores,
        max_paths=param.max_bridge_paths,
        max_hops=param.max_bridge_hops,
        query_sim=query_sim,
        community_boost=community_boost,
        use_query_scores=param.use_query_scores,
    )
    relations = _collect_unified_edges(
        graph_storage,
        hits,
        bridge_paths,
        query_sim=query_sim,
        community_boost=community_boost,
        top_k=top_k_relations,
        min_score=0.0,
        use_query_scores=param.use_query_scores,
    )
    reasoning_path = _collect_reasoning_path_edges(
        graph_storage,
        bridge_paths,
        top_k=top_k_relations,
        query_sim=query_sim,
        community_boost=community_boost,
        use_query_scores=param.use_query_scores,
    )
    sources = _collect_source_ids(entities)

    context = _render_context_sections(
        communities=communities,
        entities=entities,
        relations=relations,
        reasoning_path=reasoning_path,
        sources=sources,
        mode=param.mode,
    )
    logger.info("Building query context completed: context_length=%d chars", len(context))
    query_context = QueryContext(
        question=question,
        mode=param.mode,
        context=context,
        entities=entities,
        relations=relations,
        communities=communities,
        reasoning_path=reasoning_path,
        sources=[{"id": s} for s in sources],
        entity_hits=hits,
        local_triples=relations,
        bridge_paths=bridge_paths,
        community_reports=communities,
    )
    logger.info(
        "Query context built: mode=%s entities=%d relations=%d communities=%d bridge_edges=%d",
        param.mode,
        len(entities),
        len(relations),
        len(communities),
        len(reasoning_path),
    )
    return query_context
