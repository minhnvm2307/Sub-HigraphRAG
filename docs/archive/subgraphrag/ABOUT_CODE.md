# AI AGENT SKILLS REGISTRY

Welcome to the **SubgraphRAG AI Agent Skills Registry**. This document acts as a standardized guide for AI agents interacting with the SubgraphRAG codebase. It catalogs the available tools, execution commands, parameter structures, environment setups, and technical architectures for data processing, training, inference, and evaluation.

---

## 🛠️ Conda Environments Quick Reference

Before executing any of the skills below, ensure you activate the appropriate Conda environment:

```bash
conda activate subgraphrag
```

---

## 📂 Category 1: Data Processing & Feature Extraction

### 1. Pre-compute Entity and Relation Embeddings
Pre-computes and caches text embeddings for queries, entities, and relations to eliminate redundant transformer forward passes during retriever training and inference.

* **Command:**
  ```bash
  cd retrieve
  python emb.py -d <dataset>
  ```
* **Required Inputs & Environment:**
  * **Arguments:** `-d` / `--dataset` (`webqsp` or `cwq`).
  * **Inputs:** Pre-processed dataset files from Hugging Face (`rmanluo/RoG-cwq` or `ml1996/webqsp`) and entity identifier mapping file specified in `retrieve/configs/emb/gte-large-en-v1.5/<dataset>.yaml`.
* **Expected Outputs:**
  * Processed dataset metadata files: `retrieve/data_files/<dataset>/processed/{train,val,test}.pkl`
  * Merged PyTorch embedding tensor dictionaries: `retrieve/data_files/<dataset>/emb/gte-large-en-v1.5/{train,val,test}.pth`
* **Under the hood:**
  * Parses raw multi-hop Knowledge Graph (KG) query-subgraph samples.
  * Separates graph nodes into "text entities" (meaningful natural language names) and "non-text entities" (Freebase MIDs like `m.06w2sn5`, mapped using the identifiers list).
  * Encapsulates `gte-large-en-v1.5` within the `GTELargeEN` model class (using FP16 execution on GPU).
  * Computes 1024-dimensional dense vector embeddings for queries, text entities, and relations.
  * Saves checkpoints in shards of 200 samples (`part_*.pt`) to allow resuming if interrupted, then merges them.

### 2. Index Community Subgraphs with HiRAG
Builds hierarchical, structured graph indices for samples by partitioning the subgraph communities and generating summaries with a large language model.

* **Command:**
  ```bash
  cd retrieve
  python indexing_graph.py --dataset <dataset> --split <split> --working_dir <dir> --llm_model <model> [--force_reindex]
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:** 
    * `--dataset`: `webqsp` or `cwq`
    * `--split`: `train`, `val`, or `test`
    * `--working_dir`: Storage directory for indices (e.g., `indexed_hirag`)
    * `--llm_model`: LLM runner model ID (e.g. `ollama:llama` or `fpt` via OpenAI endpoints)
* **Expected Outputs:**
  * A community mapping configuration: `<working_dir>/<dataset>_<split>/question_graph_map.json`
  * Summarization reports stored under per-sample subdirectories.
* **Under the hood:**
  * Groups samples by unique graph topology signature (using a canonical SHA1 hash of triples).
  * Builds NetworkX digraph structures and partitions them into hierarchical semantic communities using GMM/Leiden clustering algorithms up to `semantic_max_layers` (default: 3).
  * Formulates prompt templates containing local node-relation triples and queries LLMs to synthesize human-readable "Community Reports" describing global structural context.

---

## 📈 Category 2: Retriever Model Training & Inference

### 3. Train MLP Retriever with Relation-Topic and DDE
Trains a lightweight, fast multilayer perceptron (MLP) to classify and rank triples, incorporating Directional Distance Encoding (DDE) and query-similar Relation-Topic features.

* **Command:**
  ```bash
  cd retrieve
  python train.py -d <dataset>
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:** `-d` / `--dataset` (`webqsp` or `cwq`).
  * **Configs:** Read from `retrieve/configs/retriever/<dataset>.yaml`.
  * **Inputs:** Pre-computed embeddings under `retrieve/data_files/<dataset>/emb/`.
* **Expected Outputs:**
  * A timestamped checkpoint directory (e.g., `webqsp_Mar30-17:51:30/`) containing the best model parameters file `cpt.pth`.
  * Training loss curves and validation metrics (e.g., `triple_recall@100`) logged to Wandb.
* **Under the hood:**
  * **Weak Supervision:** Computes shortest path sequences between query topic entities and target answer entities using NetworkX. Any triple on the shortest path gets a label of `1.0`.
  * **Relation-Topic Embedding Integration:** Computes cosine similarity between the query embedding $z_q$ and relation embeddings $z_r$. Relations exceeding `topic_relation_similarity_threshold` (e.g. 0.60) or matching `topic_relation_similarity_topk` (e.g. 20) are marked. If `label_topic_relation_triples` is active, triples matching these relations are assigned a weak supervision target label of `1.0`.
  * **DDE (Directional Distance Encoding):** Generates structural features by propagating entity indicators using PyG's message-passing conv layer `PEConv` along both forward and reverse directed edges over $l$ rounds.
  * **Concatenation:** Formulates a concatenated input feature vector:
    $$z_{\tau} = [z_{q} \parallel (z_{h} + PE + DDE) \parallel [z_{r} \parallel topic\_relation\_one\_hot] \parallel (z_{t} + PE + DDE)]$$
  * **MLP Scorer:** Runs a lightweight 2-layer network mapping features to a raw probability logit for each triple, trained using Binary Cross-Entropy with Logits loss (`F.binary_cross_entropy_with_logits`).

### 3b. Run Automated Retriever Benchmark (Training, Inference, and Evaluation)
Runs the entire training, inference, and evaluation pipeline sequentially for a dataset, automatically extracting metrics and saving checkpoints.

* **Command:**
  ```bash
  cd retrieve
  chmod +x run_hybrid_experiment.sh
  ./run_hybrid_experiment.sh <dataset>
  ```
* **Required Inputs & Environment:**
  * **Arguments:** `<dataset>` (`webqsp` or `cwq`).
* **Expected Outputs:**
  * Complete terminal report showing training loss, inference progress, and final recall/precision evaluation metrics.
  * Checkpoints and prediction results saved in the auto-detected directory.


### 4. Retrieve Subgraphs using Trained MLP Retriever
Executes inference on the test set using the trained MLP model to rank candidate triples and output the top-K subgraphs.

* **Command:**
  ```bash
  cd retrieve
  python inference.py -p <checkpoint_path> [--max_K <K>]
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:** 
    * `-p` / `--path`: Path to `cpt.pth` checkpoint file.
    * `--max_K`: Number of top ranked triples to retrieve (default: 500).
* **Expected Outputs:**
  * Saves a PyTorch dictionary `retrieval_result.pth` containing scored subgraphs, query entities, ground truth answers, and original path details inside the checkpoint folder.
* **Under the hood:**
  * Reconstructs the DDE and Relation-Topic input feature vectors for candidate triples.
  * Evaluates sigmoid probabilities for all candidate triples and retrieves the top-K highest-scoring triples using `torch.topk`.

### 5. Retrieve Subgraphs using Embedding Similarity (Baseline)
An embedding-only retrieval baseline that scores candidate triples using TransE-style cosine similarity without a neural classifier.

* **Command:**
  ```bash
  cd retrieve
  python inference.py -p <checkpoint_path> --embedding_only [--max_K <K>]
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:** 
    * `-p` / `--path`: Path to `cpt.pth` checkpoint file (to load embedding configurations).
    * `--embedding_only`: Flag to enable baseline.
* **Expected Outputs:**
  * Saves `retrieval_result_embedding.pth` containing scored subgraphs inside the checkpoint folder.
* **Under the hood:**
  * Clamps entity indices to safe ranges to avoid out-of-bounds errors on non-text entities.
  * Computes a TransE-style compositional vector representation:
    $$z_{\tau} = z_{h} + z_{r} + z_{t}$$
  * Ranks triples using the cosine similarity between $z_{\tau}$ and the query embedding $z_q$.

---

## 📊 Category 3: Evaluation & Analysis

### 6. Evaluate Retriever Performance (Recall and Precision)
Calculates retrieval precision and recall metrics at various K thresholds against ground-truth answers, shortest paths, and GPT annotations.

* **Command:**
  ```bash
  cd retrieve
  python eval.py -d <dataset> -p <retrieval_result_path> [--k_list <K_values>]
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:**
    * `-d` / `--dataset`: `webqsp` or `cwq`.
    * `-p` / `--path`: Path to `retrieval_result.pth` or `retrieval_result_embedding.pth`.
    * `--k_list`: Comma-separated ranks for evaluation (default: `5,10,15,20`).
* **Expected Outputs:**
  * Prints a formatted table to stdout displaying:
    * `ans_recall` & `ans_precision`
    * `shortest_path_triple_recall` & `shortest_path_triple_precision`
    * `gpt_triple_recall` & `gpt_triple_precision`
* **Under the hood:**
  * Evaluates subgraphs at each K in `k_list`.
  * Computes `ans_recall` (percentage of answers present in the retrieved node set) and `ans_precision`.
  * Computes precision and recall against shortest-path triples and GPT-extracted triples (`data_files/<dataset>/gpt_triples.pth`).

### 7. Analyze Retriever Scores and Distribution
Diagnoses the **Score Flattening** phenomenon by analyzing score distributions, differences, and entropy.

* **Command:**
  ```bash
  cd retrieve
  python analyze_result.py -p <retrieval_result_path> --analyze [--max_triples <max>]
  ```
* **Required Inputs & Environment:**
  * folder `retriever`
  * **Arguments:**
    * `-p` / `--path`: Path to `retrieval_result.pth`.
    * `--analyze`: Flag to trigger score analysis.
    * `--max_triples`: Maximum triples per query to analyze (default: 400).
* **Expected Outputs:**
  * Formatted text file `<retrieval_result_path>_readable.txt` displaying queries and retrieved triples with scores.
  * An analysis folder `<retrieval_result_path>_analysis/` containing:
    * `mean_score_vs_rank.png` (displays average score decay across ranks)
    * `score_pie_chart.png` (percentage of High, Medium, and Near-zero scores)
    * `pairwise_diff_hist.png` (distribution of $|s_i - s_j|$)
    * `entropy_hist.png` (distribution of score Shannon entropy per query)
* **Under the hood:**
  * Computes score segment percentages: High ($> 0.2$), Medium ($0.01 - 0.2$), and Near-zero ($\le 0.01$).
  * Calculates Shannon entropy and pairwise difference matrices over retrieved triples to measure diversity.

### 8. Evaluate Reasoning Path Coverage and Purity
Measures how well retrieved subgraphs cover the multi-hop reasoning paths utilized by LLMs to answer questions.

* **Command:**
  ```bash
  cd reason
  python metrics/reasoning_eval.py --samples_path <predictions_path> --retrieval_path <retrieval_path> --k_list <K_values> --output <output_path>
  ```
* **Required Inputs & Environment:**
  * folder `reasoner`
  * **Arguments:**
    * `--samples_path`: Path to prediction JSONL file containing Chain-of-Thought (CoT) text.
    * `--retrieval_path`: Path to `retrieval_result.pth` containing scored subgraphs.
    * `--k_list`: Comma-separated evaluation ranks (e.g. `10,20,50,100`).
    * `--output`: Output JSONL file path.
* **Expected Outputs:**
  * Prints average coverage and purity statistics at each specified K value.
  * Saves a JSONL file detailing matched and missing reasoning path triples for each sample.
* **Under the hood:**
  * Extracts reasoning paths from LLM generation outputs via regular expressions.
  * Compares LLM reasoning triples with retrieved top-K subgraphs.
  * Computes coverage (percentage of reasoning triples retrieved) and purity (percentage of retrieved triples that are useful for reasoning).

### 9. Generate Offline Interactive Subgraph HTML Viewer
Generates a standalone, interactive HTML browser tool to visualize and debug retrieved subgraphs.

* **Command:**
  ```bash
  cd reason
  python metrics/retrieved_graph_viewer.py --input <retrieval_result_path> --output <output_html_path> [--max_edges <max_edges>] [--num_questions <num>]
  ```
* **Required Inputs & Environment:**
  * folder `reasoner`
  * **Arguments:**
    * `--input`: Path to retrieval result `.pth` file.
    * `--output`: Target path for generated HTML file (default: `reason/metrics/retrieved_graph_viewer.html`).
    * `--max_edges`: Limit on edges shown per sample (default: 160).
    * `--num_questions`: Number of questions to export (default: 80).
* **Expected Outputs:**
  * A standalone HTML page `<output_html_path>` with embedded subgraph datasets.
* **Under the hood:**
  * Extracts subgraphs for a subset of queries.
  * Replaces the data payload token in a pre-defined HTML template containing a canvas-based force-directed layout engine.
  * Renders interactive node nodes, color-coding topic entities (green), answers (orange), shared nodes (pink), and regular nodes (blue), allowing agents or users to explore graph connections directly.

---

## 🧠 Category 4: LLM Reasoning

### 10. LLM Multi-Hop Reasoning over Subgraphs
Injects the top-K retrieved triples as text context into an LLM and runs reasoning to output final answers.

* **Command:**
  ```bash
  cd reason
  python main.py -d <dataset> --custom_retrieval <retrieval_result_path> -m <model_name> --prompt_mode <prompt_mode>
  ```
* **Required Inputs & Environment:**
  * folder `reasoner`
  * **Arguments:**
    * `-d`: `webqsp` or `cwq`.
    * `--custom_retrieval`: Path to your custom `retrieval_result.pth` file.
    * `-m`: Model name (e.g. `ollama:llama`, `gpt-4o`, `ollama:qwen3.5:9b`).
    * `--prompt_mode`: Prompt mode (e.g., `scored_100` or `scored_50` indicating the number of triples to include).
* **Expected Outputs:**
  * Output predictions saved to: `reason/results/KGQA/<dataset>/SubgraphRAG/<model_name>/<prompt_mode>-predictions.jsonl`
  * Metrics (Hit@1, macro/micro F1, macro/micro precision/recall, exact match, hallucination score) logged to Wandb.
* **Under the hood:**
  * Converts the top-K retrieved subgraph triples into text lists.
  * Injects subgraphs into formatting templates using System + CoT prompt layouts from `reason/prompts.py`.
  * Invokes the LLM generator (VLLM, local Ollama, or remote OpenAI API endpoints) to generate step-by-step reasoning.
  * Parses answers using the `ans:` prefix and computes QA metrics against ground-truth entities.

### 11. LLM Reasoning using HiRAG Community Context
Runs reasoning over the Leiden hierarchical community summaries generated in Skill 2.

* **Command:**
  ```bash
  cd reason
  python main_hirag.py --working_dir <dir> --namespace <ns> --query_mode <mode> -m <model>
  ```
* **Required Inputs & Environment:**
  * folder `reasoner`
  * **Arguments:**
    * `--working_dir`: Storage directory of HiRAG indices (e.g. `indexed_hirag`).
    * `--namespace`: Graph collection sub-namespace.
    * `--query_mode`: Query mechanism (e.g. `hi_bridge`).
    * `-m`: LLM model name.
* **Expected Outputs:**
  * Prints reasoning accuracy metrics and writes predictions.
* **Under the hood:**
  * Iterates over samples and loads HiRAG hierarchical indices.
  * Summarizes local communities and extracts global bridge node paths.
  * Prompts the LLM using community summary contexts to conduct step-by-step hierarchical inference.
