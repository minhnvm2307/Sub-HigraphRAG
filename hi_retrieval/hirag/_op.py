import re
import json
import asyncio
import networkx as nx
import time
import logging
from contextlib import contextmanager
from typing import Union
from collections import Counter, defaultdict
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._cluster_utils import Hierarchical_Clustering


@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"\033[94m[Retrieval Time: {elapsed_time:.6f} seconds]\033[0m")

def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: callable = global_config["best_model_func"]
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(        # get community reports
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(            # sort by ratings
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [                  # community reports sorted by ratings
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(    # in case community reprot is longer than token limitation
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.max_token_for_community_report,
    )
    if query_param.community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [      # the entities related to the retrieved entities
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(   # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )                               # where the source entities are the retrieved entities
    all_one_hop_nodes = set()       # find the one hop neighbors
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {               # find the text chunks of the 1-hop neighbors entities
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,     # count of relations related to the chunk
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(        # sort by relation counts
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return all_edges_data


async def _find_most_related_edges_from_paths(
    path_datas: list[dict],
    path: list[str],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # all_related_edges = await asyncio.gather(
    #     *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    # )
    # all_reasoning_path = await asyncio.gather(
    #                         *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in knowledge_graph_inst._graph.subgraph(path).edges()]
    #                     )
    all_reasoning_path = await knowledge_graph_inst.subgraph_edges(path)
    all_edges = set()
    print(all_reasoning_path)
    all_edges.update([tuple(sorted(e[:2])) for e in all_reasoning_path])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_bridge_knowledge,
    )
    return all_edges_data


# context functions
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _build_hierarchical_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    async def find_path_with_required_nodes(knowledge_graph_inst, source, target, required_nodes):
        # inital final path
        final_path = []
        # start node
        current_node = source

        # all gdb cls
        from ._storage.gdb_neo4j import Neo4jStorage
        from ._storage.gdb_networkx import NetworkXStorage
        
        # traverse the required nodes
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                if isinstance(knowledge_graph_inst, Neo4jStorage):
                    # use Neo4j's shortest_path method
                    sub_path = await knowledge_graph_inst.shortest_path(current_node, next_node)
                elif isinstance(knowledge_graph_inst, NetworkXStorage):
                    # use NetworkX's shortest_path method
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
                else:
                    # use NetworkX by default
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # merge paths (avoid adding the current node again)
            if final_path:
                final_path.extend(sub_path[1:])  # add from the second node, avoid adding the current node again
            else:
                final_path.extend(sub_path)
            
            # update the current node to the next required node
            current_node = next_node

        # finally, the path from the last required node to the target node
        try:
            if isinstance(knowledge_graph_inst, Neo4jStorage):
                # use Neo4j's shortest_path method
                sub_path = await knowledge_graph_inst.shortest_path(current_node, target)
            elif isinstance(knowledge_graph_inst, NetworkXStorage):
                # use NetworkX's shortest_path method
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            else:
                # use NetworkX by default
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # add from the second node, avoid adding the current node again
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = await find_path_with_required_nodes(knowledge_graph_inst, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        # use_reasoning_path = await _find_most_related_edges_from_entities(
        #                     path_datas, query_param, knowledge_graph_inst
        #                 )
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)
    
    # # fetch the relations of the reasoning paths
    # reasoning_path = []
    # for i in range(len(path) - 1):
    #     src = path[i]
    #     tgt = path[i + 1]
    #     cur_relation = (await knowledge_graph_inst.get_edge(src, tgt))['description']
    #     reasoning_path.append(cur_relation)
    # reasoning_path = list(set(reasoning_path))

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"].replace("\n", " ")])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # display reference info
    entities = [n["entity_name"] for n in node_datas]
    communities = [(c["level"], c["title"]) for c in use_communities]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]

    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Communities (level, cluster_id) ({len(communities)}): {communities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )

    logging.info(f"====== References ======:\n{references_context}")
    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hibridge_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    async def find_path_with_required_nodes(knowledge_graph_inst, source, target, required_nodes):
        # inital final path
        final_path = []
        current_node = source

        from ._storage.gdb_neo4j import Neo4jStorage
        from ._storage.gdb_networkx import NetworkXStorage
        
        for next_node in required_nodes:
            try:
                if isinstance(knowledge_graph_inst, Neo4jStorage):
                    sub_path = await knowledge_graph_inst.shortest_path(current_node, next_node)
                elif isinstance(knowledge_graph_inst, NetworkXStorage):
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
                else:
                    sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            if final_path:
                final_path.extend(sub_path[1:])
            else:
                final_path.extend(sub_path)
            
            current_node = next_node

        try:
            if isinstance(knowledge_graph_inst, Neo4jStorage):
                sub_path = await knowledge_graph_inst.shortest_path(current_node, target)
            elif isinstance(knowledge_graph_inst, NetworkXStorage):
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            else:
                sub_path = nx.shortest_path(knowledge_graph_inst._graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    # find some top-k entities in each communities in use_communities
    key_entities = []
    max_entity_num = query_param.top_m
    if use_communities:
        for c in use_communities:
            cur_community_key_entities = []
            community_entities = c['nodes']
            # find the top-k entities in this community
            cur_community_key_entities.extend(
                [e for e in overall_node_datas if e['entity_name'] in community_entities][:max_entity_num]
            )
            key_entities.append(cur_community_key_entities)
    else:
        key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = await find_path_with_required_nodes(knowledge_graph_inst, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_higlobal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_communities = await _find_most_related_community_from_entities(     # related communities
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )


    logger.info(
        f"Using {len(use_communities)} communities, {len(use_text_units)} text units"
    )
    
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Backgrounds-----
```csv
{communities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hilocal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    relation_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relation_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relation_context = list_of_list_to_csv(relation_section_list)
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relation_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


# query functions
async def hierarchical_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hierarchical_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def hierarchical_bridge_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hibridge_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def hierarchical_local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_hilocal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def hierarchical_global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_higlobal_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            community_reports,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
