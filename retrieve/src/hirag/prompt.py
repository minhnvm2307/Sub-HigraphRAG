GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {
    "META_ENTITY_TYPES": [
        "organization",
        "person",
        "location",
        "event",
        "product",
        "technology",
        "industry",
        "mathematics",
        "social sciences",
    ],
    "DEFAULT_TUPLE_DELIMITER": "<|>",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
}

PROMPTS["summary_clusters"] = """You are tasked with analyzing a set of entity descriptions and a given list of meta attributes. Your goal is to summarize at least one attribute entity for the entity set in the given entity descriptions. And the summarized attribute entity must match the type of at least one meta attribute in the given meta attribute list. The relationship between the entity set and the generated attribute entity should be clear and logical.

-Steps-
1. Identify at least one attribute entity for the given entity description list. For each attribute entity, extract:
- entity_name: capitalized
- entity_type: one of [{meta_attribute_list}]
- entity_description: comprehensive description
Format:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From each given entity, identify all pairs of (source_entity, target_entity) that are clearly related to the attribute entities identified in step 1. There should be no relations between attribute entities.
Format:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return all entities and relationships as one list, separated by {record_delimiter}
4. End with {completion_delimiter}

Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}
Output:
"""

PROMPTS["community_report"] = """You are an AI assistant that writes a grounded report for a graph community.

Return a JSON object with:
{
  "title": "...",
  "summary": "...",
  "rating": 0-10 float,
  "rating_explanation": "...",
  "findings": [{"summary": "...", "explanation": "..."}]
}

Only use the provided evidence.

Input:
{input_text}

Output:
"""
