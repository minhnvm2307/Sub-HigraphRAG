import asyncio
from collections import defaultdict
from .base import BaseGraphStorage, CommunitySchema
from ._storage import JsonKVStorage
from .prompt import PROMPTS
from ._utils import logger, pack_user_ass_to_openai_messages


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: dict,
    max_token_size: int = 12000,
    global_config: dict = {},
) -> str:
    """
    Create a textual description of a single community, including its nodes,
    edges, and connections to other communities.
    """
    nodes = community.get("nodes", [])
    edges = community.get("edges", [])
    level = community.get("level", 0)
    title = community.get("title", "")

    # Basic community info
    community_describe = f"Community ({title}) at level {level} has {len(nodes)} nodes and {len(edges)} edges.\n"

    # Node details
    node_describes = []
    for node_id in nodes:
        node_data = await knwoledge_graph_inst.get_node(node_id)
        if node_data:
            node_describes.append(
                f"- {node_id} ({node_data.get('entity_type', 'N/A')}): {node_data.get('description', '')}"
            )
    community_describe += "\nNodes:\n" + "\n".join(node_describes)

    # Edge details
    edge_describes = []
    for src, tgt in edges:
        edge_data = await knwoledge_graph_inst.get_edge(src, tgt)
        if edge_data:
            edge_describes.append(f"- ({src} -> {tgt}): {edge_data.get('description', '')}")
    community_describe += "\nEdges:\n" + "\n".join(edge_describes)

    return community_describe


async def generate_community_report(
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> JsonKVStorage:
    """
    Generates structured reports for each community in the graph using an LLM.
    """
    use_llm_func: callable = global_config["best_model_func"]
    community_report_kv = JsonKVStorage(
        namespace=f"hi_retrieval/HiRAG/results/{knwoledge_graph_inst.namespace}_community_reports",
        global_config=global_config,
    )

    community_schema = await knwoledge_graph_inst.community_schema()
    if not community_schema:
        logger.warning("No community schema found in the graph. Skipping report generation.")
        return community_report_kv

    all_reports = {}
    tasks = []

    for community_id, community_data in community_schema.items():
        task = _generate_single_report(
            community_id, community_data, knwoledge_graph_inst, use_llm_func, global_config
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for report in results:
        if report:
            all_reports.update(report)

    if all_reports:
        await community_report_kv.upsert(all_reports)

    return community_report_kv


async def _generate_single_report(
    community_id: str,
    community_data: dict,
    knwoledge_graph_inst: BaseGraphStorage,
    use_llm_func: callable,
    global_config: dict,
) -> dict:
    """Helper to generate a report for a single community."""
    try:
        community_text_description = await _pack_single_community_describe(
            knwoledge_graph_inst, community_data, global_config=global_config
        )

        prompt_template = PROMPTS["community_report"]
        use_prompt = prompt_template.format(
            community_description=community_text_description,
            response_format="json_object"
        )
        
        # The actual LLM call to generate the report
        llm_messages = pack_user_ass_to_openai_messages(use_prompt, "")
        raw_response = await use_llm_func(messages=llm_messages)
        
        # Assuming the response is a JSON string in the content
        response_content = raw_response.choices[0].message.content
        report_json = json.loads(response_content)


        final_report = CommunitySchema(
            **community_data,
            report_string=str(report_json), # Or a formatted string version
            report_json=report_json,
        )
        return {community_id: final_report}
    except Exception as e:
        logger.error(f"Failed to generate report for community {community_id}: {e}")
        return {}
