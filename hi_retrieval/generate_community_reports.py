import argparse
import asyncio
from dataclasses import asdict
import torch

from reason.llm_utils import llm_init
from hirag.subgraph_rag_graph import SubgraphRAGGraph
from hirag.community_builder import generate_community_report
from hirag.hirag import HiRAG  # To get default configs
from retrieve.src.model.text_encoders.gte_large_en import GTELargeEN


async def main():
    parser = argparse.ArgumentParser(description="Generate community reports for SubgraphRAG graphs.")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp", help="Dataset name (e.g., webqsp, cwq)")
    parser.add_argument("-m", "--model_name", type=str, default="gpt-4o-mini", help="Model name for LLM")
    parser.add_argument("--split", type=str, default="test", help="Data split (e.g., test)")
    parser.add_argument("--retrieval_file", type=str, default="retrieval_result.pth", help="Name of the retrieval result file.")
    args = parser.parse_args()

    # Use HiRAG dataclass to get default configurations
    config = HiRAG()
    global_config = asdict(config)

    # Initialize LLM
    llm = llm_init(args.model_name)
    global_config["best_model_func"] = llm

    # Initialize Embedding Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = GTELargeEN(device=device)
    global_config["embedding_model"] = embedding_model

    # 1. Load the graph from SubgraphRAG's output
    print("Loading graph from SubgraphRAG retrieval data...")
    graph_loader = SubgraphRAGGraph(
        dataset_name=args.dataset_name,
        split=args.split,
        retrieval_file=args.retrieval_file,
        global_config=global_config
    )

    # 2. Perform clustering
    print("Performing Leiden clustering on the graph...")
    await graph_loader.clustering("leiden")

    # 3. Generate community reports
    print("Generating community reports...")
    community_reports_kv = await generate_community_report(graph_loader, global_config)

    # 4. Save the reports
    output_path = f"hi_retrieval/HiRAG/results/{args.dataset_name}_community_reports.json"
    print(f"Saving community reports to {output_path}...")
    await community_reports_kv.index_done_callback()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
