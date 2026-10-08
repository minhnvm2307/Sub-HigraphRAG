# Stage 2: Reasoning

## Table of Contents

* [Installation](#installation)
* [Pre-processed Results for Reproducibility](#pre-processed-results-for-reproducibility)
* [Inference with LLMs](#inference-with-llms)

## Installation

```bash
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5 openai==1.50.2 wandb
```

## Reasoning (Inference)

### Using Pre-Processed Retrieval Results for Reproducibility

We provide pre-processed results for reproducibility of the paper experiments. To download them

```bash
hf download siqim311/SubgraphRAG --revision main --local-dir ./
```

- `scored_triples` stores the pre-processed retrieval results.
- `results/KGQA` stores the reasoning results.

After downloading the pre-processed results, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
python main.py -d cwq --prompt_mode scored_100
```

### Using Alternative Retrieval Results

To use your custom retrieval results (e.g., from your own retriever),

```bash
python main.py -d webqsp --prompt_mode scored_50 -m ollama:llama
```

where `--custom_retrieval` points to your `retrieval_result.pth` file. This will:
- Load your custom retrieved triples directly
- Skip the need for pre-processed RoG predictions
- Use the subgraph data from Hugging Face
- Run the reasoning module on top of your retrieved triples

For other models, you can also specify the model name:
```bash
python main.py -d webqsp --prompt_mode scored_100 --custom_retrieval ../retrieve/webqsp_Mar28-18:25:22/retrieval_result.pth -m "gpt-4o"
```

Or with ollama:
```bash
python main.py -d webqsp --prompt_mode scored_100 --custom_retrieval ../retrieve/webqsp_Mar30-17:51:30/retrieval_result.pth -m "ollama:qwen3.5:9b"
```

### Reasoning Path Coverage Evaluation

To measure how much of the reasoning path is covered by the top-K retrieved triples,
run the evaluator on result file that includes the reasoning-path text. If the
sample file does not already contain structured scored triples, pass the retrieval
result path as well.

```bash
python metrics/reasoning_eval.py \
	--samples_path ./sample_predictions.jsonl \
	--retrieval_path ./scored_triples/retrieval_result.pth \
	--k_list 10,20,50,100 \
	--output ./reasoning_eval_details.jsonl
```

The script prints coverage and purity at each K, plus a JSONL file with the
matched and missing reasoning triples for each sample.

### Reasoning using HiRAG (main_hirag.py)
```bash
python main_hirag.py --working_dir ../retrieve/indexed_hirag --namespace webqsp_indexed_graph --num_samples 1000 --query_mode hi_bridge -m ollama:llama --log_level DEBUG 2>&1 | tee reason.log 
```

### Config

Our used config for each dataset can be found in `./config`.
