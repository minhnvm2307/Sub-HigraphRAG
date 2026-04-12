"""
build_index.py

Pipeline (per topic-entity group):
  1. Load all groups from SubgraphRAG .pth retrieval file.
  2. For each group: Leiden clustering → community reports → persist.

No entity/relation extraction — graph already exists in the .pth file.

Usage
-----
# Full run
python build_index.py -d webqsp --split test -m gpt-4o-mini

# Resume: skip groups that already have a community_reports JSON
python build_index.py -d webqsp --split test --skip_if_cached

# Only specific topic entities (debug / partial run)
python build_index.py -d webqsp --split test --filter "Jamaica" "Barack Obama"

# Use GMM clustering instead of Leiden
python build_index.py -d webqsp --split test --cluster_algo gmm
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict

import torch

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------


def _import_or_die(module_path: str, install_hint: str = ""):
    import importlib
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Cannot import '{module_path}': {e}. {install_hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build HiRAG index from SubgraphRAG retrieval data (per topic-entity group)."
    )
    p.add_argument("-d", "--dataset_name", default="webqsp")
    p.add_argument("--split", default="test")
    p.add_argument("-r", "--retrieval_file", default="retrieval_result.pth")
    p.add_argument("--retrieval_dir", default="reason/scored_triples",
                   help="Folder containing the .pth file")
    p.add_argument("-w", "--working_dir", default=None,
                   help="Cache dir (default: ./hirag_cache_<dataset>_<split>)")
    p.add_argument("-m", "--model_name", default="gpt-4o-mini")
    p.add_argument("--cluster_algo", default="leiden", choices=["leiden", "gmm"],
                   help="Clustering algorithm")
    p.add_argument("--max_cluster_size", type=int, default=10)
    p.add_argument("--skip_if_cached", action="store_true",
                   help="Skip groups whose community_reports JSON already exists")
    p.add_argument("--filter", nargs="+", dest="topic_entity_filter", default=None,
                   metavar="ENTITY",
                   help="Only process these topic entities (space-separated)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-group pipeline
# ---------------------------------------------------------------------------

async def process_group(
    topic_entity: str,
    graph,                      # SubgraphRAGGraph instance
    community_reports_kv,       # JsonKVStorage (per-group)
    global_config: dict,
    cluster_algo: str,
    skip_if_cached: bool,
    gmm_cfg=None,
):
    """Run clustering + community report for one topic-entity group."""

    # Check cache
    reports_json = community_reports_kv._file_name   # JsonKVStorage exposes this
    if skip_if_cached and os.path.exists(reports_json):
        logger.info(f"  [SKIP] '{topic_entity}' — reports already cached at {reports_json}")
        return False   # False = skipped

    n_nodes = graph._graph.number_of_nodes()
    if n_nodes == 0:
        logger.warning(f"  [SKIP] '{topic_entity}' — graph is empty")
        return False

    logger.info(f"  [Cluster] '{topic_entity}' ({n_nodes} nodes, algo={cluster_algo})")

    # Clustering
    try:
        if cluster_algo == "gmm":
            from hirag.gmm_clustering import register_gmm_clustering, GMMClusterConfig
            register_gmm_clustering(graph, cfg=gmm_cfg or GMMClusterConfig(),
                                    global_config=global_config)
        await graph.clustering(cluster_algo)
    except Exception as e:
        logger.error(f"  [ERROR] Clustering failed for '{topic_entity}': {e}")
        logger.debug("", exc_info=True)
        return False

    # Validate clustering
    clustered = sum(1 for _, d in graph._graph.nodes(data=True) if "clusters" in d)
    logger.info(f"  [Cluster] '{topic_entity}' — {clustered}/{n_nodes} nodes clustered")
    if clustered == 0:
        logger.warning(
            f"  [WARN] No clusters for '{topic_entity}'. "
            "Graph may be too small or disconnected. Try --max_cluster_size smaller."
        )
        return False

    # Persist clustered graph (.graphml)
    await graph.index_done_callback()

    # Community reports
    logger.info(f"  [Report] Generating community reports for '{topic_entity}' …")
    try:
        from hirag.community_builder import generate_community_report
        await generate_community_report(community_reports_kv, graph, global_config)
        await community_reports_kv.index_done_callback()
    except Exception as e:
        logger.error(f"  [ERROR] Community report failed for '{topic_entity}': {e}")
        logger.debug("", exc_info=True)
        return False

    logger.info(f"  [Done] '{topic_entity}' ✓")
    return True   # True = processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    working_dir = args.working_dir or f"./hirag_cache_{args.dataset_name}_{args.split}"
    os.makedirs(working_dir, exist_ok=True)
    logger.info(f"Working directory: {os.path.abspath(working_dir)}")

    # ------------------------------------------------------------------ #
    # 1. Build global_config                                               #
    # We only need the keys relevant to clustering + community reports.   #
    # ------------------------------------------------------------------ #
    try:
        from hirag.hirag import HiRAG
        # Get raw dataclass field defaults WITHOUT triggering __post_init__
        import dataclasses
        hirag_defaults = {f.name: f.default_factory() if callable(f.default_factory) else f.default
                          for f in dataclasses.fields(HiRAG)
                          if f.default is not dataclasses.MISSING or
                             f.default_factory is not dataclasses.MISSING}
    except Exception:
        hirag_defaults = {}

    global_config: dict = {
        **hirag_defaults,
        "working_dir": working_dir,
        "max_graph_cluster_size": args.max_cluster_size,
        "graph_cluster_seed": 0xDEADBEEF,
        "best_model_func": None,
        "cheap_model_func": None,
        "convert_response_to_json_func": None,
        "addon_params": {},
        "enable_local": False,
        "enable_naive_rag": False,
    }

    # ------------------------------------------------------------------ #
    # 2. LLM                                                               #
    # ------------------------------------------------------------------ #
    logger.info(f"Initialising LLM: {args.model_name}")
    try:
        from reason.llm_utils import llm_init
        from hirag._utils import convert_response_to_json
        llm = llm_init(args.model_name)
        global_config["best_model_func"] = llm
        global_config["cheap_model_func"] = llm
        global_config["convert_response_to_json_func"] = convert_response_to_json
    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        raise

    # ------------------------------------------------------------------ #
    # 3. Load all group graphs                                             #
    # ------------------------------------------------------------------ #
    from hirag.subgraph_rag_graph import SubgraphRAGGraph, GraphLoadConfig

    load_cfg = GraphLoadConfig(retrieval_dir=args.retrieval_dir)

    logger.info("=" * 60)
    logger.info("STEP 1 — Loading graphs")
    logger.info("=" * 60)
    graphs = SubgraphRAGGraph.load_all_groups(
        retrieval_file=args.retrieval_file,
        dataset_name=args.dataset_name,
        split=args.split,
        global_config=global_config,
        load_cfg=load_cfg,
        topic_entity_filter=args.topic_entity_filter,
    )

    if not graphs:
        logger.error("No graphs loaded — check retrieval file and filter settings.")
        sys.exit(1)

    # Print a few summaries for sanity check
    for entity, graph in list(graphs.items())[:3]:
        graph.print_summary()

    # ------------------------------------------------------------------ #
    # 4. Cluster + community reports (per group)                           #
    # ------------------------------------------------------------------ #
    from hirag._storage import JsonKVStorage

    logger.info("=" * 60)
    logger.info("STEP 2 — Clustering + Community Reports")
    logger.info("=" * 60)

    total = len(graphs)
    n_done = n_skipped = n_failed = 0

    for idx, (entity, graph) in enumerate(graphs.items(), 1):
        logger.info(f"[{idx}/{total}] Processing: '{entity}'")

        # Each group gets its own KV store so reports don't collide
        # Namespace: community_reports_<safe_entity>
        from hirag.subgraph_rag_graph import _safe_name
        group_namespace = f"community_reports_{args.dataset_name}_{args.split}_{_safe_name(entity)}"
        community_reports_kv = JsonKVStorage(
            namespace=group_namespace,
            global_config=global_config,
        )

        result = await process_group(
            topic_entity=entity,
            graph=graph,
            community_reports_kv=community_reports_kv,
            global_config=global_config,
            cluster_algo=args.cluster_algo,
            skip_if_cached=args.skip_if_cached,
        )

        if result is True:
            n_done += 1
        elif result is False:
            # Check if it was skipped or failed
            reports_json = community_reports_kv._file_name
            if os.path.exists(reports_json):
                n_skipped += 1
            else:
                n_failed += 1

    # ------------------------------------------------------------------ #
    # 5. Merge all community reports into one summary JSON (optional)      #
    # ------------------------------------------------------------------ #
    summary_path = os.path.join(
        working_dir,
        f"{args.dataset_name}_{args.split}_all_community_reports.json"
    )
    merged = {}
    for fname in os.listdir(working_dir):
        if fname.startswith(f"kv_store_community_reports_{args.dataset_name}") and fname.endswith(".json"):
            fpath = os.path.join(working_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    merged.update(json.load(f))
            except Exception as e:
                logger.warning(f"Could not read {fpath}: {e}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info(
        f"\n{'='*60}\n"
        f"  DONE\n"
        f"  Total groups : {total}\n"
        f"  Processed    : {n_done}\n"
        f"  Skipped      : {n_skipped}\n"
        f"  Failed       : {n_failed}\n"
        f"  Merged report: {summary_path} ({len(merged)} communities)\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))