import os
import json
import wandb
import argparse
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data, get_data_from_custom_retrieval
from llm_utils import llm_init_langchain
from prompt_manager import PromptManager
from qa_graph import create_qa_graph
from main import eval_all, load_checkpoint, save_checkpoint

def setup_monitoring_callbacks(enable_langsmith=False, sample_rate=0.0):
    """
    Sets up monitoring callbacks for LangSmith with sampling and disable support.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    tracing_env = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1")
    if not enable_langsmith and not tracing_env and sample_rate <= 0.0:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("[Monitoring] LangSmith tracing is DISABLED (quota protected).")
        return None

    # Map LANGSMITH_API_KEY to LANGCHAIN_API_KEY for standard tracing
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
        
    tracer = None
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Prevent untargeted global auto-tracing
        project_name = os.environ.get("LANGCHAIN_PROJECT", "SubgraphRAG-QA")
        print(f"[Monitoring] LangSmith tracing enabled with sampling rate: {sample_rate:.1%}. Project: {project_name}")
        
        try:
            from langchain_core.tracers import LangChainTracer
            tracer = LangChainTracer(project_name=project_name)
            print("[Monitoring] LangChainTracer callback initialized successfully.")
        except ImportError:
            print("[Monitoring] Warning: langchain-core not installed. Tracing disabled.")
    else:
        print("[Monitoring] Warning: LangChain API key not found. Tracing is disabled.")
        
    return tracer

def main():
    parser = argparse.ArgumentParser(description="LangGraph-based RAG for KGQA Evaluation")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str, help="Path to pre-computed triple scores")
    parser.add_argument("--custom_retrieval", type=str, help="Path to custom retrieval results (retrieval_result.pth)")
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="ollama:llama", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples to process (for quick testing)")
    parser.add_argument("--enable_langsmith", action="store_true", help="Explicitly enable LangSmith tracing")
    parser.add_argument("--langsmith_sample_rate", type=float, default=0.0, help="Sampling rate for LangSmith tracing (0.0=disabled, 1.0=all)")
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    use_custom_retrieval = args.custom_retrieval is not None
    
    if not use_custom_retrieval:
        pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    
    print("--- DEBUG: Starting main function ---")
    run_name = f"langgraph-{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}"
    if use_custom_retrieval:
        run_name += "-custom_retrieval"
        
    # Start Wandb Run
    print("--- DEBUG: Initializing wandb ---")
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)
    print("--- DEBUG: Wandb initialized successfully ---")

    # Resolve score dict path if not custom retrieval
    if args.score_dict_path is None and not use_custom_retrieval:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    # Define output files path
    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    if use_custom_retrieval:
        custom_name = Path(args.custom_retrieval).parent.name
        raw_pred_file_path = raw_pred_folder_path / f"langgraph-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-custom_{custom_name}-predictions.jsonl"
    else:
        raw_pred_file_path = raw_pred_folder_path / f"langgraph-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions.jsonl"

    # Setup Tracing / Monitoring Callbacks
    tracer = setup_monitoring_callbacks(args.enable_langsmith, args.langsmith_sample_rate)

    # Initialize prompt manager and LLM
    prompt_manager = PromptManager()
    llm = llm_init_langchain(model_name, temperature=temperature, max_tokens=max_tokens)
    
    # Load dataset
    if use_custom_retrieval:
        print(f"Loading custom retrieval results from: {args.custom_retrieval}")
        data = get_data_from_custom_retrieval(dataset_name, args.custom_retrieval, split, prompt_mode)
    else:
        data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)

    # Restrict samples size if requested
    if args.num_samples is not None:
        print(f"Restricting evaluation dataset to the first {args.num_samples} samples.")
        data = data[:args.num_samples]

    # Prime PromptManager to resolve prompts once and get correct version
    prompt_manager.get_prompts(prompt_mode, model_name, llm_mode)
    prompt_version = prompt_manager.get_prompt_version()
    print(f"Using prompt version: {prompt_version}")

    # Create & Compile LangGraph
    print("Compiling LangGraph workflow...")
    graph = create_qa_graph(llm, prompt_manager, prompt_mode, llm_mode, model_name, thres)

    # Resume check
    start_idx = len(load_checkpoint(raw_pred_file_path))
    print(f"Resuming from index: {start_idx}")
    
    if start_idx >= len(data):
        print("All items have already been processed.")
    else:
        with open(raw_pred_file_path, "a") as pred_file:
            for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
                # Build initial graph state inputs
                inputs = {
                    "id": each_qa["id"],
                    "question": each_qa["question"],
                    "scored_triplets": each_qa["scored_triplets"],
                    "a_entity": each_qa.get("a_entity", []),
                    "sys_prompt": "",
                    "cot_prompt": "",
                    "sys_query": "",
                    "user_query": "",
                    "all_query": "",
                    "cot_query": "",
                    "raw_response": "",
                    "prediction": "",
                    "supports": [],
                    "hit": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                }
                
                # Build run configuration for LangSmith tracing (with sampling)
                sample_callbacks = []
                if tracer is not None and (args.langsmith_sample_rate >= 1.0 or random.random() < args.langsmith_sample_rate):
                    sample_callbacks = [tracer]

                run_config = {
                    "callbacks": sample_callbacks,
                    "metadata": {
                        "prompt_mode": prompt_mode,
                        "llm_mode": llm_mode,
                        "model_name": model_name,
                        "thres": thres,
                        "prompt_version": prompt_version,
                        "sample_id": each_qa["id"],
                    },
                    "tags": [prompt_mode, llm_mode, f"thres_{thres}"]
                }
                
                # Execute LangGraph node flow
                try:
                    final_state = graph.invoke(inputs, config=run_config)
                    prediction = final_state["prediction"]
                    supports = final_state["supports"]
                except Exception as e:
                    print(f"Error processing sample {each_qa['id']}: {e}")
                    prediction = "ans: not available"
                    supports = []
                
                # Format to save checkpoint
                save_qa = {
                    "id": each_qa["id"],
                    "question": each_qa["question"],
                    "prediction": prediction,
                    "a_entity": each_qa.get("a_entity", []),
                    "supports": supports
                }
                save_checkpoint(pred_file, save_qa)

    # Finalize predictions and report evaluation metrics
    print(f"Predictions saved to: {raw_pred_file_path}")
    eval_all(raw_pred_file_path, run, subset=True)
    wandb.finish()

if __name__ == "__main__":
    main()
