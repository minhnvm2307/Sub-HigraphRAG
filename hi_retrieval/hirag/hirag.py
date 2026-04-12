import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
import torch

from retrieve.src.model.text_encoders.gte_large_en import GTELargeEN

from ._op import (
    # chunking_by_token_size,
    # extract_entities,
    # extract_hierarchical_entities,
    # get_chunks,
    hierarchical_query,
    hierarchical_bridge_query,
    hierarchical_local_query,
    hierarchical_global_query,
    # hierarchical_nobridge_query,
    # naive_query,
)
from ._storage import (
    JsonKVStorage,
    # NanoVectorDBStorage,
    NetworkXStorage,
    # Neo4jStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from .community_builder import generate_community_report


@dataclass
class HiRAG:
    working_dir: str = field(
        default_factory=lambda: f"./hirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True

    # text chunking
    chunk_func: Callable = None
    chunk_token_size: int = 8192 # Max for GTE
    chunk_overlap_token_size: int = 100

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1024, # GTE Large is 1024
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_model: object = None # Will be initialized to GTELargeEN
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    best_model_func: callable = None # Will be set from reason.llm_utils
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_func: callable = None # Will be set from reason.llm_utils
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8

    # entity extraction
    entity_extraction_func: callable = None
    hierarchical_entity_extraction_func: callable = None

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = None
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HiRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Initialize the embedding model
        if self.embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = GTELargeEN(device=device)
        
        # The embedding_func is now a method of the initialized model
        self.embedding_func = self.embedding_model.embed

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        if self.best_model_func:
            self.best_model_func = limit_async_func_call(self.best_model_max_async)(
                partial(self.best_model_func, hashing_kv=self.llm_response_cache)
            )
        if self.cheap_model_func:
            self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
                partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
            )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode in ["hi", "hi_nobridge", "hi_bridge", "hi_local", "hi_global"] and not self.enable_hierachical_mode:
            raise ValueError(f"enable_hierachical_mode is False, cannot query in {param.mode} mode")

        query_map = {
            "hi": hierarchical_query,
            "hi_bridge": hierarchical_bridge_query,
            "hi_local": hierarchical_local_query,
            "hi_global": hierarchical_global_query,
            # "hi_nobridge": hierarchical_nobridge_query,
            # "naive": naive_query,
        }

        query_func = query_map.get(param.mode)
        
        if query_func:
            if param.mode == "naive":
                response = await query_func(
                    query,
                    self.chunks_vdb,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
            else:
                response = await query_func(
                    query,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.community_reports,
                    self.text_chunks,
                    param,
                    asdict(self),
                )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            # inserting_chunks = get_chunks(
            #     new_docs=new_docs,
            #     chunk_func=self.chunk_func,
            #     tokenizer=self.embedding_model.tokenizer,
            #     overlap_token_size=self.chunk_overlap_token_size,
            #     max_token_size=self.chunk_token_size,
            # )
            inserting_chunks = None

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            extraction_func = self.hierarchical_entity_extraction_func if self.enable_hierachical_mode else self.entity_extraction_func
            logger.info(f"\033[94m[{'Hierarchical ' if self.enable_hierachical_mode else ''}Entity Extraction]...\033[0m")
            
            maybe_new_kg = await extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )

            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            
            # ---------- update clusterings of graph
            logger.info("\033[94m[Community Report]...\033[0m")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
