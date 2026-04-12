from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from sklearn.mixture import GaussianMixture
import networkx as nx
import torch

from .base import CommunityReport, EntityHit, QueryContext, QueryParam

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


def semantic_cluster_layers(graph_storage, vector_index, max_layers: int = 3, min_cluster_size: int = 3) -> list[dict]:
    if vector_index.embeddings.numel() == 0 or len(vector_index.ids) < min_cluster_size:
        logger.info(
            "Skip semantic clustering: entities=%d min_cluster_size=%d",
            len(vector_index.ids),
            min_cluster_size,
        )
        return []

    current_ids = vector_index.ids[:]
    current_embeddings = vector_index.embeddings.clone()
    layers = []
    logger.info(
        "Start semantic clustering: entities=%d max_layers=%d min_cluster_size=%d",
        len(current_ids),
        max_layers,
        min_cluster_size,
    )

    for layer_idx in range(max_layers):
        if len(current_ids) < min_cluster_size:
            logger.info(
                "Stop semantic clustering at layer=%d due to too few entities=%d",
                layer_idx,
                len(current_ids),
            )
            break
        max_k = min(max(2, len(current_ids) // 2), len(current_ids) - 1)
        if max_k < 2:
            logger.info("Stop semantic clustering at layer=%d because max_k < 2", layer_idx)
            break

        best_model = None
        best_bic = None
        for k in range(2, max_k + 1):
            model = GaussianMixture(n_components=k, random_state=0, n_init=3)
            model.fit(current_embeddings.numpy())
            bic = model.bic(current_embeddings.numpy())
            logger.debug("Semantic clustering layer=%d k=%d bic=%.4f", layer_idx, k, bic)
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_model = model

        if best_model is None:
            logger.warning("Semantic clustering failed to fit a model at layer=%d", layer_idx)
            break

        labels = best_model.predict(current_embeddings.numpy()).tolist()
        groups = defaultdict(list)
        for node_id, label in zip(current_ids, labels):
            groups[int(label)].append(node_id)
        logger.info(
            "Semantic clustering layer=%d selected_components=%d candidate_groups=%d",
            layer_idx,
            best_model.n_components,
            len(groups),
        )

        next_ids = []
        next_embeddings = []
        layer_nodes = []
        for cluster_id, members in groups.items():
            if len(members) < min_cluster_size:
                logger.debug(
                    "Skip semantic cluster: layer=%d cluster=%d size=%d",
                    layer_idx,
                    cluster_id,
                    len(members),
                )
                continue
            meta_node = f"cluster:L{layer_idx}:C{cluster_id}"
            member_indices = [current_ids.index(node_id) for node_id in members]
            meta_embedding = current_embeddings[member_indices].mean(dim=0)
            graph_storage.upsert_node(
                meta_node,
                id=meta_node,
                entity_name=meta_node,
                entity_type="SEMANTIC_CLUSTER",
                description=_summarize_members(graph_storage, members),
                members=json.dumps(sorted(members)),
                layer=int(layer_idx),
            )
            for member in members:
                graph_storage.upsert_edge(
                    meta_node,
                    member,
                    id=f"{meta_node}|semantic_parent|{member}",
                    relation="semantic_parent",
                    description=f"{meta_node} groups {member}",
                    weight=1.0,
                )
            next_ids.append(meta_node)
            next_embeddings.append(meta_embedding)
            layer_nodes.append({"cluster_node": meta_node, "members": sorted(members)})

        if not next_ids:
            logger.info("Semantic clustering produced no valid clusters at layer=%d", layer_idx)
            break

        vector_index.upsert(
            (node_id, emb, {"entity_name": node_id, "semantic_layer": layer_idx})
            for node_id, emb in zip(next_ids, next_embeddings)
        )
        layers.append({"layer": layer_idx, "clusters": layer_nodes})
        logger.info(
            "Semantic clustering layer=%d completed: new_cluster_nodes=%d",
            layer_idx,
            len(next_ids),
        )
        current_ids = next_ids
        current_embeddings = torch.stack(next_embeddings, dim=0)

    logger.info("Semantic clustering finished: layers=%d", len(layers))
    return layers


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


def build_community_reports(graph_storage, communities: dict[str, dict]) -> dict[str, dict]:
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
        summary = (
            f"Community {community_id} contains {len(node_names)} nodes and "
            f"{len(payload['edges'])} edges. Top entities: {_summarize_members(graph_storage, node_names)}. "
            f"Top relations: {', '.join(rel for rel, _ in relation_counts.most_common(5))}."
        )
        reports[community_id] = {
            "community_id": community_id,
            "level": payload["level"],
            "title": f"Community {community_id}",
            "summary": summary,
            "node_ids": node_names,
            "edge_ids": payload["edges"],
            "rating": float(len(payload["edges"])),
            "metadata": {"relation_counts": dict(relation_counts)},
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
