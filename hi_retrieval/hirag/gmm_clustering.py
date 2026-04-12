"""
hirag/gmm_clustering.py

Drop-in GMM-based clustering that produces the same `clusters` node
attribute format that NetworkXStorage.community_schema() expects.

How to use
----------
Option A — monkey-patch into an existing SubgraphRAGGraph:

    from hirag.gmm_clustering import register_gmm_clustering
    register_gmm_clustering(graph)          # adds "gmm" to _clustering_algorithms
    await graph.clustering("gmm")

Option B — call directly:

    from hirag.gmm_clustering import gmm_cluster_graph
    await gmm_cluster_graph(graph, global_config)

GMM Tuning knobs
----------------
All parameters are in GMMClusterConfig below.  Change them in one place.

Algorithm overview
------------------
1. Embed each node using node2vec (same as Leiden path).
2. Reduce dimension with UMAP.
3. Fit a Gaussian Mixture Model; use BIC to choose the number of components.
4. Assign every node to the cluster with the highest posterior probability
   (hard assignment; soft threshold can be enabled via `soft_threshold`).
5. Write results as JSON into node attribute "clusters" — same format as
   Leiden, so community_schema() and all downstream code works unchanged.

Only flat (single-level) clustering is implemented.  Set level=0.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GMMClusterConfig:
    """
    All tuneable knobs for GMM clustering.

    n_components_range : (min, max) range to search for optimal k.
                         Larger graphs may need a higher upper bound.
    umap_dim           : target dimensionality for UMAP reduction.
    umap_n_neighbors   : neighbourhood size for UMAP.  Try 5–30.
    umap_metric        : distance metric used by UMAP.
    bic_rel_tol        : early-stop BIC search when improvement < this fraction.
    gmm_n_init         : how many GMM initialisations to average (higher = more stable).
    gmm_random_state   : reproducibility seed.
    soft_threshold     : if > 0, a node can belong to multiple clusters whose
                         posterior prob exceeds this value.  Set to 0 for hard
                         (single-cluster) assignment.
    level              : community level written into the "clusters" attribute.
                         Keep 0 for single-level; the downstream code handles it.
    node2vec_dims      : dimension of node2vec embedding before UMAP reduction.
    node2vec_walk_len  : walk length for node2vec.
    node2vec_num_walks : number of walks per node.
    node2vec_window    : context window for node2vec skip-gram.
    node2vec_seed      : reproducibility seed for node2vec.
    """
    n_components_range: tuple = (2, 50)
    umap_dim: int = 10
    umap_n_neighbors: int = 15
    umap_metric: str = "cosine"
    bic_rel_tol: float = 1e-3
    gmm_n_init: int = 5
    gmm_random_state: int = 42
    soft_threshold: float = 0.0          # 0 = hard assignment
    level: int = 0                       # community level label
    # node2vec params
    node2vec_dims: int = 64
    node2vec_walk_len: int = 30
    node2vec_num_walks: int = 10
    node2vec_window: int = 2
    node2vec_seed: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_with_node2vec(G, cfg: GMMClusterConfig) -> tuple[np.ndarray, list]:
    """
    Returns (embeddings, node_ids) using graspologic's node2vec.
    Raises ImportError if graspologic is not installed.
    """
    try:
        from graspologic.embed import node2vec_embed
    except ImportError:
        raise ImportError(
            "graspologic is required for node2vec embedding. "
            "Install with: pip install graspologic"
        )

    logger.info(
        f"[GMM] Running node2vec — dims={cfg.node2vec_dims}, "
        f"walks={cfg.node2vec_num_walks}, walk_len={cfg.node2vec_walk_len}"
    )
    embeddings, nodes = node2vec_embed(
        G,
        dimensions=cfg.node2vec_dims,
        walk_length=cfg.node2vec_walk_len,
        num_walks=cfg.node2vec_num_walks,
        window_size=cfg.node2vec_window,
        random_seed=cfg.node2vec_seed,
    )
    node_ids = [G.nodes[n].get("id", n) for n in nodes]
    logger.info(f"[GMM] node2vec complete — embedding shape: {embeddings.shape}")
    return embeddings, nodes, node_ids  # nodes = raw nx node keys


def _umap_reduce(embeddings: np.ndarray, cfg: GMMClusterConfig) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        )

    dim = min(cfg.umap_dim, len(embeddings) - 2)
    if dim < 2:
        logger.warning(
            f"[GMM] Too few nodes ({len(embeddings)}) for UMAP reduction. "
            "Using raw embeddings."
        )
        return embeddings

    n_neighbors = min(cfg.umap_n_neighbors, len(embeddings) - 1)
    logger.info(
        f"[GMM] UMAP reduction — target_dim={dim}, n_neighbors={n_neighbors}, "
        f"metric={cfg.umap_metric}"
    )
    reducer = umap.UMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        metric=cfg.umap_metric,
        random_state=cfg.gmm_random_state,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info(f"[GMM] UMAP complete — reduced shape: {reduced.shape}")
    return reduced


def _optimal_gmm(
    embeddings: np.ndarray, cfg: GMMClusterConfig
) -> "GaussianMixture":
    from sklearn.mixture import GaussianMixture

    lo, hi = cfg.n_components_range
    hi = min(hi, len(embeddings) - 1)
    lo = max(lo, 2)

    best_bic = float("inf")
    best_k = lo
    prev_bic = float("inf")

    logger.info(f"[GMM] BIC search over k ∈ [{lo}, {hi}]")
    for k in range(lo, hi + 1):
        gm = GaussianMixture(
            n_components=k,
            n_init=cfg.gmm_n_init,
            init_params="k-means++",
            random_state=cfg.gmm_random_state,
        )
        gm.fit(embeddings)
        bic = gm.bic(embeddings)
        logger.debug(f"  k={k:3d}  BIC={bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k

        # Early-stop if BIC improvement is negligible
        if prev_bic != float("inf"):
            rel_change = abs(prev_bic - bic) / (abs(prev_bic) + 1e-10)
            if rel_change < cfg.bic_rel_tol:
                logger.info(
                    f"[GMM] Early-stop BIC search at k={k} "
                    f"(rel_change={rel_change:.2e} < tol={cfg.bic_rel_tol})"
                )
                break
        prev_bic = bic

    logger.info(f"[GMM] Optimal k = {best_k}  (BIC = {best_bic:.2f})")

    final_gm = GaussianMixture(
        n_components=best_k,
        n_init=cfg.gmm_n_init,
        init_params="k-means++",
        random_state=cfg.gmm_random_state,
    )
    final_gm.fit(embeddings)
    return final_gm


def _assign_clusters(
    gm, embeddings: np.ndarray, cfg: GMMClusterConfig
) -> List[List[int]]:
    """
    Returns a list of cluster-id lists, one per embedding.
    Soft mode: a node belongs to every cluster where posterior > threshold.
    Hard mode (threshold==0): single best cluster per node.
    """
    probs = gm.predict_proba(embeddings)
    if cfg.soft_threshold > 0:
        assignments = [
            list(np.where(p > cfg.soft_threshold)[0]) for p in probs
        ]
        # Ensure every node has at least one cluster
        for i, a in enumerate(assignments):
            if not a:
                assignments[i] = [int(np.argmax(probs[i]))]
    else:
        assignments = [[int(np.argmax(p))] for p in probs]
    return assignments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def gmm_cluster_graph(graph, global_config: dict, cfg: Optional[GMMClusterConfig] = None):
    """
    Cluster `graph._graph` with GMM and write results into node attributes.

    Parameters
    ----------
    graph        : SubgraphRAGGraph (or any NetworkXStorage subclass)
    global_config: dict — same dict passed to SubgraphRAGGraph
    cfg          : GMMClusterConfig — all tuning knobs
    """
    cfg = cfg or GMMClusterConfig()
    G = graph._graph

    if G.number_of_nodes() < 3:
        logger.error(
            f"[GMM] Graph has only {G.number_of_nodes()} nodes — too few to cluster."
        )
        return

    logger.info(
        f"[GMM] Starting GMM clustering on graph with "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # 1. Node embeddings via node2vec
    try:
        raw_embeddings, nx_nodes, node_ids = _embed_with_node2vec(G, cfg)
    except Exception as e:
        logger.exception(f"[GMM] node2vec embedding failed: {e}")
        raise

    # 2. UMAP dimensionality reduction
    try:
        reduced = _umap_reduce(raw_embeddings, cfg)
    except Exception as e:
        logger.exception(f"[GMM] UMAP reduction failed: {e}")
        raise

    # 3. Fit GMM + BIC search
    try:
        gm = _optimal_gmm(reduced, cfg)
    except Exception as e:
        logger.exception(f"[GMM] GMM fitting failed: {e}")
        raise

    # 4. Assign clusters
    assignments = _assign_clusters(gm, reduced, cfg)

    # 5. Write into graph node attributes (same format as Leiden)
    for nx_node, cluster_ids in zip(nx_nodes, assignments):
        cluster_list = [
            {"level": cfg.level, "cluster": int(cid)} for cid in cluster_ids
        ]
        G.nodes[nx_node]["clusters"] = json.dumps(cluster_list)

    # Stats
    from collections import Counter
    flat_ids = [cid for a in assignments for cid in a]
    counts = Counter(flat_ids)
    logger.info(
        f"[GMM] Done — {gm.n_components} clusters, "
        f"avg size={np.mean(list(counts.values())):.1f}, "
        f"min={min(counts.values())}, max={max(counts.values())}"
    )


def register_gmm_clustering(
    graph,
    cfg: Optional[GMMClusterConfig] = None,
    global_config: Optional[dict] = None,
):
    """
    Inject 'gmm' into graph._clustering_algorithms so you can call:
        await graph.clustering("gmm")

    Parameters
    ----------
    graph        : SubgraphRAGGraph (or any NetworkXStorage subclass)
    cfg          : GMMClusterConfig  (uses defaults if None)
    global_config: dict  (falls back to graph.global_config if None)
    """
    import asyncio

    _cfg = cfg or GMMClusterConfig()
    _global_config = global_config or getattr(graph, "global_config", {})

    async def _gmm_clustering_impl():
        await gmm_cluster_graph(graph, _global_config, _cfg)

    graph._clustering_algorithms["gmm"] = _gmm_clustering_impl
    logger.info(
        "[GMM] Registered 'gmm' clustering algorithm on graph. "
        "Call: await graph.clustering('gmm')"
    )