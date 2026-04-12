from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from dataclasses import asdict

from retrieve.src.config.retriever import load_yaml
from retrieve.src.dataset.retriever import RetrieverDataset
from retrieve.src.hirag import HiRAG, QueryParam
from retrieve.src.hirag.hirag import setup_logger


def main(args) -> None:
    setup_logger(args.log_level)

    config_file = args.config_file or f"retrieve/configs/retriever/{args.dataset}.yaml"
    config = load_yaml(config_file)

    dataset = RetrieverDataset(
        config=config,
        split=args.split,
        skip_no_path=args.skip_no_path,
    )
    samples = dataset.processed_dict_list[: args.num_samples]
    if not samples:
        raise ValueError("No samples available for testing")

    namespace = args.namespace or f"{args.dataset}_{args.split}_{args.num_samples}"
    hirag = HiRAG(
        working_dir=args.working_dir,
        namespace=namespace,
        semantic_max_layers=args.semantic_max_layers,
        semantic_min_cluster_size=args.semantic_min_cluster_size,
        log_level=args.log_level,
    )

    stats = hirag.index_samples(samples)
    print("Index stats:")
    print(json.dumps(stats, indent=2))

    query_param = QueryParam(
        mode=args.mode,
        top_k_entities=args.top_k_entities,
        top_k_triples=args.top_k_triples,
        top_k_communities=args.top_k_communities,
        max_bridge_paths=args.max_bridge_paths,
        max_bridge_hops=args.max_bridge_hops,
        community_level=args.community_level,
    )
    context = hirag.query_sample(samples[0], param=query_param)

    print("\nFirst sample query:")
    print(samples[0]["question"])
    print("\nQuery context:")
    print(json.dumps(asdict(context), indent=2))
    print(f"\nArtifacts written to: {os.path.join(args.working_dir, namespace)}")


if __name__ == "__main__":
    parser = ArgumentParser("HiRAG small-sample debug runner")
    parser.add_argument("--dataset", type=str, default="cwq", choices=["cwq", "webqsp"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default="retrieve/debug_hirag")
    parser.add_argument("--namespace", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--mode", type=str, default="hi", choices=["hi", "hi_local", "hi_global", "hi_bridge"])
    parser.add_argument("--top_k_entities", type=int, default=8)
    parser.add_argument("--top_k_triples", type=int, default=20)
    parser.add_argument("--top_k_communities", type=int, default=5)
    parser.add_argument("--max_bridge_paths", type=int, default=5)
    parser.add_argument("--max_bridge_hops", type=int, default=4)
    parser.add_argument("--community_level", type=int, default=0)
    parser.add_argument("--semantic_max_layers", type=int, default=3)
    parser.add_argument("--semantic_min_cluster_size", type=int, default=3)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--skip_no_path", action="store_true")
    main(parser.parse_args())
