# HiRAG Graph Clustering, Hierarchical Query, and MLP Integration Plan

# MAIN GOAL
Your task is to find the way to intergrate method 1' s code to retrieve folder of method 2 so that it indexing and retrieve on the dataset schema (dataset that been embedding using gte_large)
Refactor all the code so that hi_rag compatible with main graphrag method (method 2)
Remain only need code, clean the code by use less files as possible 
The desired code should be in retrieve.src.hirag/
with the code of storage and graphload -> graphml files
code to build indexing (GMM layers and Leidein + community reports layers) -> store
code like _op.py to intergrate method 1 and method 2 (local + global) to answer the answer of dataset

## 1. Purpose of this document

This file is a transfer note for reusing the current HiRAG graph-retrieval design in another project.

The current project has two different retrieval methods:

Methods 1: hi_retrieval/ (retrievel by indexing the graph by GMM and Community reports using leidein algorithm)

1. Semantic hierarchical clustering during entity extraction
   - Implemented with embedding-based clustering and GMM-style grouping.
   - Used to create higher-level summarized entities/relations.

2. Structural graph community clustering after graph construction
   - Implemented with Leiden on the built graph.
   - Used to generate community reports for query-time global context.

The hierarchical query pipeline then combines:

- retrieved seed entities,
- community summaries,
- shortest-path bridge knowledge,
- source text chunks.

This is the main logic to preserve when moving to another repository (Method 2).

Method 2: retrieve/ (the main methods that you must not change any):

- This load datafiles and emb entity and relation of dataset that have question -> graph and answer with 1 graph to many question
- Build MLP model to predict triples scores and ranking to choose top K to get subgraph related to topic entities


# Some summary of the core of method 1

## 2. Core modules and responsibilities

### Main orchestration

- `hirag/hirag.py`
  - Main class: `HiRAG`
  - Initializes storage, embedding function, LLM functions, graph DB, KV stores, and vector DBs.
  - Main entrypoints:
    - `insert()` / `ainsert()`
    - `query()` / `aquery()`

### Indexing and retrieval operations

- `hirag/_op.py`
  - Chunking
  - Entity and relation extraction
  - Hierarchical entity extraction
  - Community report generation
  - Query-context builders for `hi`, `hi_bridge`, `hi_local`, `hi_global`, `hi_nobridge`, and `naive`

### Hierarchical semantic clustering

- `hirag/_cluster_utils.py`
  - UMAP reduction
  - Gaussian Mixture selection by BIC
  - Iterative clustering + summary generation across layers
  - Produces abstracted entities/relations that are inserted back into the graph

### Graph storage and Leiden clustering

- `hirag/_storage/gdb_networkx.py`
  - Graph load/save
  - Node/edge upsert
  - Leiden clustering over built graph
  - Community schema generation
  - Subgraph edge extraction for reasoning path context

### Data contracts

- `hirag/base.py`
  - `QueryParam`
  - `BaseGraphStorage`
  - `BaseVectorStorage`
  - `BaseKVStorage`

### Example runtime wiring

- `hi_Search_custom.py`
  - Shows how provider-specific embedding + LLM functions are injected into `HiRAG`

## 3. End-to-end indexing flow

### Step 1: document ingestion and chunking

In `HiRAG.ainsert()`:

1. Raw docs are hashed and stored in `full_docs`.
2. `get_chunks()` in `hirag/_op.py` splits docs into token chunks.
3. Chunks are stored in `text_chunks`.
4. If naive RAG is enabled, chunks are also embedded into `chunks_vdb`.

Important output stores:

- `kv_store_full_docs.json`
- `kv_store_text_chunks.json`
- optionally `vdb_chunks.json`

### Step 2: hierarchical entity/relation extraction

If `enable_hierachical_mode=True`, `HiRAG.ainsert()` calls:

- `extract_hierarchical_entities()` in `hirag/_op.py`

This flow is:

1. Run LLM entity extraction chunk by chunk.
2. Run optional gleaning loop if the model says more entities remain.
3. Build `all_entities` from chunk outputs.
4. Embed extracted entities using `entity_vdb.embedding_func`.
5. Run relation extraction chunk by chunk using extracted chunk-local entities as context.
6. Run hierarchical clustering over extracted entities.
7. Merge original + clustered entities/relations into the graph.
8. Upsert final entity records into the entity vector DB.

Important detail:

- Hierarchical extraction is not only “cluster labels”.
- It creates new summarized semantic nodes/edges using LLM summarization.
- These new nodes are later stored in the same graph as the original extracted entities.

## 4. Hierarchical semantic clustering logic

Implemented in `hirag/_cluster_utils.py` by `Hierarchical_Clustering.perform_clustering()`.

### Actual logic

1. Input entities already have embeddings.
2. UMAP reduces embeddings.
3. GMM cluster count is selected using BIC.
4. Each layer groups current entities into clusters.
5. For each cluster:
   - if only one item, keep it as-is;
   - if too large, randomly downsample until prompt length is manageable;
   - summarize cluster members with LLM;
   - parse the summary output back into entity/relation records;
   - embed the newly summarized entities;
   - use them as the next layer’s clustering input.
6. Stop when:
   - cluster sparsity exceeds threshold, or
   - sparsity improvement becomes too small, or
   - deduplicated entity count becomes too small.

### What this produces

The return value is a list of layers:

- layer 0: original extracted entities
- later layers: summarized meta-entities/meta-relations

These layers are later flattened into:

- `maybe_nodes`
- `maybe_edges`

and inserted into the graph in `extract_hierarchical_entities()`.

### Why this matters for migration

If you port this method to another project, preserve these four assumptions:

1. Every entity record must keep `entity_name`, `entity_type`, `description`, `source_id`.
2. Every clustered summary must be parsed back to the same schema as normal extraction.
3. New summarized entities must be embedded again before the next layer.
4. Original and summarized entities must live in one graph, not separate stores.

## 5. Graph construction and Leiden community layer

After hierarchical extraction finishes, `HiRAG.ainsert()` calls:

1. `self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)`
2. `generate_community_report(...)`

### Leiden clustering

In `hirag/_storage/gdb_networkx.py`:

- `_leiden_clustering()` uses `graspologic.partition.hierarchical_leiden`
- Each graph node receives a `clusters` field like:
  - `[{ "level": ..., "cluster": ... }, ...]`

This clustering is graph-structural, not embedding-semantic.

### Community report generation

In `hirag/_op.py`, `generate_community_report()`:

1. Reads graph community schema from storage.
2. Builds a textual description of each community.
3. Calls LLM to generate structured community reports.
4. Stores:
   - `report_string`
   - `report_json`
   - original community metadata

Important output store:

- `kv_store_community_reports.json`

## 6. Query-time hierarchical retrieval flow

Query entrypoint is `HiRAG.aquery()` in `hirag/hirag.py`.

Supported modes:

- `hi`
- `hi_bridge`
- `hi_local`
- `hi_global`
- `hi_nobridge`
- `naive`

### Common first step in all hierarchical modes

Most hierarchical query builders in `hirag/_op.py` begin with:

- `entities_vdb.query(query, top_k=...)`

This means the current system uses vector similarity over entity embeddings as the seed retrieval step.

### `hi` mode

`_build_hierarchical_query_context()` combines:

1. top candidate entities
2. most related community reports
3. shortest-path bridge relations between key entities
4. supporting source chunks

Returned context sections:

- Backgrounds
- Reasoning Path
- Detail Entity Information
- Source Documents

This is the main “full hierarchical” mode.

### `hi_bridge` mode

Focuses on:

- community context
- bridge path knowledge
- source documents

Useful when path-based reasoning is more important than rich local relation context.

### `hi_local` mode

Focuses on:

- entities
- direct relations
- source chunks

Useful for neighborhood-level retrieval around seed entities.

### `hi_global` mode

Focuses on:

- community reports
- source chunks

Useful when summary-level graph context matters more than local graph traversal.

### `hi_nobridge` mode

Uses:

- community reports
- entities
- relations
- source chunks

but excludes the shortest-path bridge reasoning block.

## 7. Query-time subgraph logic

The most important subgraph behavior is in `hirag/_op.py`:

1. Retrieve seed entities from `entities_vdb`.
2. Expand to communities through node `clusters`.
3. Select key entities per community.
4. Find shortest path through selected entities.
   - Uses `NetworkXStorage` or `Neo4jStorage`
5. Convert subgraph/path edges into prompt context.
6. Pull source chunks connected to seed entities and one-hop neighbors.

So the practical query-time subgraph is:

- not a generic graph neighborhood dump,
- but a focused subgraph built from:
  - seed entities,
  - community membership,
  - shortest-path bridging,
  - chunk provenance.

## 8. Files and artifacts worth reusing in another project

If moving this method to another codebase, these are the highest-priority files to port or replicate:

### Must keep

- `hirag/hirag.py`
- `hirag/_op.py`
- `hirag/_cluster_utils.py`
- `hirag/base.py`

### Storage layer

- `hirag/_storage/gdb_networkx.py` or `hirag/_storage/gdb_neo4j.py`
- `hirag/_storage/vdb_nanovectordb.py`
- `hirag/_storage/kv_json.py`

### Prompt layer

- `hirag/prompt.py`

Without compatible prompts, extraction, summary, and community generation will not behave the same.

## 9. Recommended migration abstraction

For reuse in another project, separate the system into 4 replaceable layers:

1. Document preprocessing
   - chunking

2. Graph indexing
   - entity extraction
   - relation extraction
   - hierarchical semantic clustering
   - graph insertion
   - Leiden clustering
   - community reports

3. Candidate retrieval
   - current implementation: `entities_vdb.query()`
   - future implementation: MLP retriever or hybrid retriever

4. Context assembly
   - local relations
   - bridge path
   - community reports
   - source chunks

This is the cleanest place to integrate another retrieval method without rewriting the whole graph pipeline.

## 10. MLP retriever integration plan

## Recommendation

Do not replace the graph-building pipeline first.

The lowest-risk integration is:

1. keep current indexing and graph construction,
2. replace or augment only the seed entity retrieval stage,
3. let the existing hierarchical query and subgraph-building logic stay unchanged.

This keeps:

- GMM hierarchical summarization
- Leiden communities
- shortest-path bridge retrieval
- chunk provenance logic

while changing only how seed entities are chosen.

## 11. Best insertion point for MLP

Today the main retrieval seam is:

- `entities_vdb.query(query, top_k=...)`

inside:

- `_build_local_query_context()`
- `_build_hierarchical_query_context()`
- `_build_hibridge_query_context()`
- `_build_higlobal_query_context()`
- `_build_hilocal_query_context()`

Recommended architecture:

Create one retrieval abstraction, for example:

- `BaseEntityRetriever`
- `VectorEntityRetriever`
- `MLPEntityRetriever`
- `HybridEntityRetriever`

Then the query builders should call:

- `entity_retriever.query(query, top_k=...)`

instead of calling `entities_vdb` directly.

## 12. Suggested MLP integration strategies

### Option A: MLP reranker on top of vector recall

Recommended first implementation.

Flow:

1. use vector DB to recall top 50 to 200 candidate entities
2. score `(query, entity)` pairs with MLP
3. rerank and keep top_k
4. continue with existing graph expansion

Why this is best first:

- low code risk
- no need to train full-corpus exhaustive scorer first
- avoids replacing current embedding search infra
- easy to A/B test against current behavior

### Option B: MLP as primary retriever

Flow:

1. precompute entity features
2. MLP scores entities directly for a query
3. choose top_k
4. continue with graph expansion

Risk:

- slower unless you build an efficient candidate-generation stage
- usually still needs a recall stage, so pure replacement is harder

### Option C: Hybrid score

Use a weighted score such as:

- `final_score = a * vector_similarity + b * mlp_score + c * graph_prior`

Possible graph priors:

- node degree
- community match
- relation density
- path availability

This is the strongest medium-term design, but not the best first milestone.

## 13. Concrete implementation plan

### Phase 1: isolate retrieval dependency

Goal:

- remove direct dependency on `entities_vdb.query()` from query builders

Tasks:

1. Add a retriever interface in `hirag/base.py` or a new module.
2. Wrap current vector search into `VectorEntityRetriever`.
3. Update `HiRAG` to initialize `self.entity_retriever`.
4. Replace direct calls in hierarchical query builders.

Definition of done:

- current behavior unchanged when using vector retriever

### Phase 2: add MLP reranker

Goal:

- use MLP to rerank vector-recalled candidates

Tasks:

1. Add a new module, for example `hirag/_retrieval/mlp_retriever.py`
2. Input features for each `(query, entity)` pair:
   - query embedding
   - entity embedding
   - cosine similarity
   - optional lexical overlap
   - optional node degree / cluster count
3. Return ranked entities in the same shape currently expected:
   - `[{ "entity_name": ..., "score": ... }]`

Definition of done:

- same downstream pipeline works with no changes
- query modes `hi`, `hi_local`, `hi_global`, `hi_bridge` still function

### Phase 3: evaluate hybrid retrieval

Goal:

- compare:
  - vector only
  - vector + MLP rerank
  - hybrid weighted scoring

Metrics:

- answer quality
- retrieved entity precision
- supporting chunk relevance
- path usefulness
- latency

### Phase 4: optional graph-aware MLP expansion

Only after Phase 2 is stable.

Possible extensions:

1. use MLP to rank communities after seed retrieval
2. use MLP to rank candidate bridge nodes or edges
3. use MLP to rank source chunks before prompt assembly

## 14. Product/PM view: scope, risk, and dependency

### Scope recommendation

Initial integration scope should be query-time only.

Do not mix these changes in one iteration:

- new MLP retriever
- graph schema redesign
- prompt redesign
- chunking redesign
- new clustering algorithm

That would make debugging very hard.

### Main risks

1. Schema drift
   - query builders expect entity names that exist in graph storage

2. Candidate mismatch
   - MLP may return entities not present in the graph if indexing is not aligned

3. Latency increase
   - reranking too many candidates per query can be expensive

4. Evaluation ambiguity
   - better seed retrieval does not always mean better final answer if path/community assembly is weak

### Dependency checklist

Before integrating MLP in another project, confirm:

1. entity embeddings are available
2. entity names are stable IDs across vector store and graph store
3. graph nodes still contain:
   - `description`
   - `source_id`
   - `clusters`
4. community reports exist or can be regenerated
5. shortest-path graph functions exist in the target graph backend

## 15. Practical recommendation for the next implementation

If the next project already has its own graph storage and chunk pipeline, port the logic in this order:

1. query context builders from `hirag/_op.py`
2. retrieval abstraction replacing direct `entities_vdb.query()`
3. MLP reranker
4. only then port the hierarchical extraction and GMM summarization layer if needed

If the next project needs full HiRAG behavior, then port in this order:

1. chunking
2. entity/relation extraction
3. hierarchical semantic clustering with summary
4. graph insertion
5. Leiden community reports
6. query context builders
7. MLP reranker

## 16. Short conclusion

The current project’s core design is:

- semantic hierarchical abstraction by GMM-style clustering + LLM summary,
- structural community abstraction by Leiden,
- query-time subgraph assembly from retrieved entities, communities, bridge paths, and source chunks.

For MLP integration, the best first move is not to replace the graph pipeline.

The best first move is to replace the seed entity retrieval step with:

- vector recall + MLP reranking,

while preserving the existing hierarchical context assembly logic.
