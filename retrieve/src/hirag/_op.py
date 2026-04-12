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
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not available, using raw embeddings for clustering")
        return embeddings
    if len(embeddings) <= 2:
        return embeddings
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
    if max_clusters <= 2:
        return 1
    n_clusters = np.arange(1, max_clusters)
    bics = []
    prev_bic = float("inf")
    for n in n_clusters:
        bic = _fit_gaussian_mixture(int(n), embeddings, random_state)
        bics.append(bic)
        if prev_bic != float("inf") and abs(prev_bic - bic) / (abs(prev_bic) + 1e-8) < rel_tol:
            break
        prev_bic = bic
    return int(n_clusters[int(np.argmin(bics))])


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

    try:
        from graspologic.partition import hierarchical_leiden

        logger.info("Run structural community detection with hierarchical_leiden")
        community_rows = hierarchical_leiden(
            nx.Graph(graph),
            max_cluster_size=max(10, graph.number_of_nodes() // 5),
            random_seed=0,
        )
        node_clusters = defaultdict(list)
        for row in community_rows:
            node_clusters[row.node].append({"level": int(row.level), "cluster": str(row.cluster)})
    except ImportError:
        logger.warning("graspologic not available, falling back to greedy_modularity_communities")
        communities = list(nx.community.greedy_modularity_communities(nx.Graph(graph)))
        node_clusters = defaultdict(list)
        for idx, community in enumerate(communities):
            for node_id in community:
                node_clusters[node_id].append({"level": 0, "cluster": str(idx)})

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
        "Structural community detection finished: communities=%d nodes=%d",
        len(result),
        len(node_clusters),
    )
    return result


def build_community_reports(graph_storage, communities: dict[str, dict], llm_callable=None) -> dict[str, dict]:
    reports = {}
    logger.info("Build community reports: communities=%d", len(communities))
    for community_id, payload in communities.items():
        node_names = payload["nodes"]
        edge_labels = []
        for edge_id in payload["edges"]:
            src, tgt = edge_id.split("|", 1)
            data = graph_storage.graph.edges[src, tgt]
            edge_labels.append(data.get("relation", "related_to"))
        relation_counts = Counter(edge_labels)
        rows = [["id", "entity", "type", "description"]]
        for idx, node_name in enumerate(node_names):
            node_data = graph_storage.graph.nodes[node_name]
            rows.append([idx, node_name, node_data.get("entity_type", "UNKNOWN"), node_data.get("description", "")])
        relation_rows = [["id", "source", "target", "relation", "description"]]
        for idx, edge_id in enumerate(payload["edges"]):
            src, tgt = edge_id.split("|", 1)
            edge_data = graph_storage.graph.edges[src, tgt]
            relation_rows.append([idx, src, tgt, edge_data.get("relation", "related_to"), edge_data.get("description", "")])
        community_input = f"Entities:\n```csv\n{list_of_list_to_csv(rows)}\n```\nRelationships:\n```csv\n{list_of_list_to_csv(relation_rows)}\n```"
        report_json = None
        if llm_callable is not None:
            report_json = extract_first_complete_json(llm_callable(PROMPTS["community_report"].format(input_text=community_input)))
        if report_json is None:
            summary = (
                f"Community {community_id} contains {len(node_names)} nodes and "
                f"{len(payload['edges'])} edges. Top entities: {_summarize_members(graph_storage, node_names)}. "
                f"Top relations: {', '.join(rel for rel, _ in relation_counts.most_common(5))}."
            )
            report_json = {
                "title": f"Community {community_id}",
                "summary": summary,
                "rating": float(min(10.0, len(payload["edges"]))),
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
            "rating": float(report_json.get("rating", len(payload["edges"]))),
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
    logger.debug("Entity query finished: top_k=%d hits=%d", top_k, len(hits))
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


def build_query_context(graph_storage, vector_index, community_store, question: str, query_embedding: torch.Tensor, param: QueryParam) -> QueryContext:
    logger.info(
        "Build query context: mode=%s top_k_entities=%d top_k_triples=%d top_k_communities=%d",
        param.mode,
        param.top_k_entities,
        param.top_k_triples,
        param.top_k_communities,
    )
    hits = _entity_hits(graph_storage, vector_index, query_embedding, top_k=param.top_k_entities)
    context = QueryContext(question=question, mode=param.mode, entity_hits=hits)
    if param.mode in {"hi", "hi_local"}:
        context.local_triples = _local_context(graph_storage, hits, top_k_triples=param.top_k_triples)
    if param.mode in {"hi", "hi_global"}:
        context.community_reports = _global_context(
            hits,
            community_store,
            community_level=param.community_level,
            top_k_communities=param.top_k_communities,
        )
    if param.mode in {"hi", "hi_bridge"}:
        context.bridge_paths = _bridge_context(
            graph_storage,
            hits,
            max_paths=param.max_bridge_paths,
            max_hops=param.max_bridge_hops,
        )
    logger.info(
        "Query context built: mode=%s entity_hits=%d local_triples=%d communities=%d bridge_paths=%d",
        param.mode,
        len(context.entity_hits),
        len(context.local_triples),
        len(context.community_reports),
        len(context.bridge_paths),
    )
    return context
