from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from preprocess.prepare_prompts import get_prompts
from llm_utils import llm_inf_all_langchain, parse_response

class GraphQAState(TypedDict):
    id: str
    question: str
    scored_triplets: List[tuple]
    a_entity: List[str]
    sys_prompt: str
    cot_prompt: str
    sys_query: str
    user_query: str
    all_query: str
    cot_query: str
    raw_response: str
    prediction: str
    supports: List[str]
    # Evaluation Metrics
    hit: float
    f1: float
    precision: float
    recall: float

def create_qa_graph(llm, prompt_manager, prompt_mode: str, llm_mode: str, model_name: str, thres: float):
    """
    Creates and compiles a LangGraph StateGraph for the GraphRAG QA Reasoning stage.
    """
    
    def resolve_prompts_node(state: GraphQAState) -> Dict[str, Any]:
        """Node 1: Resolves prompt templates from manager/hub/registry."""
        sys_prompt, cot_prompt = prompt_manager.get_prompts(prompt_mode, model_name, llm_mode)
        return {
            "sys_prompt": sys_prompt,
            "cot_prompt": cot_prompt
        }

    def build_prompt_node(state: GraphQAState) -> Dict[str, Any]:
        """Node 2: Formats triplet context and constructs final LLM inputs."""
        temp_qa = {
            "question": state["question"],
            "scored_triplets": state["scored_triplets"],
            "a_entity": state["a_entity"],
            "good_paths_rog": [],
            "good_triplets_rog": []
        }
        # Generate queries based on prompt mode and thresholds
        formatted = get_prompts(temp_qa, prompt_mode, state["sys_prompt"], state["cot_prompt"], thres)
        return {
            "sys_query": formatted["sys_query"],
            "user_query": formatted["user_query"],
            "all_query": formatted["all_query"],
            "cot_query": formatted["cot_query"]
        }

    def llm_reason_node(state: GraphQAState) -> Dict[str, Any]:
        """Node 3: Executes LLM reasoning under target configuration."""
        sample = {
            "sys_query": state["sys_query"],
            "user_query": state["user_query"],
            "all_query": state["all_query"],
            "cot_query": state["cot_query"]
        }
        res = llm_inf_all_langchain(llm, sample, llm_mode, model_name)
        return {
            "raw_response": res
        }

    def parse_response_node(state: GraphQAState) -> Dict[str, Any]:
        """Node 4: Parses structural prediction and supporting facts."""
        pred, supports = parse_response(state["raw_response"])
        return {
            "prediction": pred,
            "supports": supports
        }

    def evaluate_response_node(state: GraphQAState) -> Dict[str, Any]:
        """Node 5: Evaluates the parsed response against ground truth entities."""
        from metrics.evaluate_results_corrected import get_pred, eval_precision, eval_recall, eval_f1, eval_hit
        
        question = state["question"]
        prediction = state["prediction"]
        answer = sorted(list(set(state["a_entity"])), key=len, reverse=True)
        
        # Clean answer entities for when / what year questions
        if 'when' in question.lower() or 'what year' in question.lower():
            for idx in range(len(answer)):
                if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                    answer[idx] = answer[idx].split('-')[0]
                    
        double_check = any([keyword in question.lower() for keyword in ['when', 'what year', 'which year', 'where', 'sport', "what countr", "language", 'nba finals', 'world series']])
        
        # prediction is already parsed by parse_response, which returns a string where candidate answers start with "ans:"
        pred_list = get_pred(prediction, None)
        
        precision_score, matched_1, num_pred = eval_precision(pred_list, answer, double_check)
        recall_score, matched_2, num_answer = eval_recall(pred_list, answer, double_check)
        f1_score = eval_f1(precision_score, recall_score)
        hit_score = eval_hit(pred_list, answer, double_check)
        
        return {
            "hit": float(hit_score),
            "f1": float(f1_score),
            "precision": float(precision_score),
            "recall": float(recall_score)
        }

    # Initialize state workflow
    workflow = StateGraph(GraphQAState)
    
    # Add Nodes
    workflow.add_node("resolve_prompts", resolve_prompts_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("llm_reason", llm_reason_node)
    workflow.add_node("parse_response", parse_response_node)
    workflow.add_node("evaluate_response", evaluate_response_node)

    # Define Graph flow
    workflow.add_edge(START, "resolve_prompts")
    workflow.add_edge("resolve_prompts", "build_prompt")
    workflow.add_edge("build_prompt", "llm_reason")
    workflow.add_edge("llm_reason", "parse_response")
    workflow.add_edge("parse_response", "evaluate_response")
    workflow.add_edge("evaluate_response", END)

    # Compile the graph
    return workflow.compile()
