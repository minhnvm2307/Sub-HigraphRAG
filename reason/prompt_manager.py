import os
from typing import Tuple

class PromptManager:
    """
    Manages prompt templates for the GraphRAG reasoning stage.
    Supports fetching versioned prompts from LangSmith registry,
    with automatic fallback to local definitions.
    """
    def __init__(self):
        self.use_langsmith = os.environ.get("LANGCHAIN_API_KEY") is not None
        self.langsmith_client = None
        self.last_fetched_version = "unknown"
        self._cache = {}
        
        if self.use_langsmith:
            try:
                from langsmith import Client
                self.langsmith_client = Client()
            except ImportError:
                self.use_langsmith = False

    def get_prompt_version(self) -> str:
        """Returns the version of the last successfully loaded prompt set."""
        return self.last_fetched_version

    def get_local_prompts(self, prompt_mode: str, model_name: str, llm_mode: str) -> Tuple[str, str]:
        """Loads default prompts from local prompts.py file."""
        self.last_fetched_version = "local"
        if 'gpt' in model_name or 'gpt' in prompt_mode:
            if 'gptLabel' in prompt_mode:
                from prompts import sys_prompt_gpt, cot_prompt_gpt
                return sys_prompt_gpt, cot_prompt_gpt
            else:
                from prompts import icl_sys_prompt, icl_cot_prompt
                return icl_sys_prompt, icl_cot_prompt
        elif 'noevi' in prompt_mode:
            from prompts import noevi_sys_prompt, noevi_cot_prompt
            return noevi_sys_prompt, noevi_cot_prompt
        elif 'icl' in llm_mode:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
        elif 'custom' in llm_mode:
            from prompts import custom_sys_prompt, custom_cot_prompt
            return custom_sys_prompt, custom_cot_prompt
        else:
            from prompts import sys_prompt, cot_prompt
            return sys_prompt, cot_prompt

    def _resolve_prompts(self, prompt_mode: str, model_name: str, llm_mode: str) -> Tuple[str, str]:
        """Actual resolution logic from registry or local."""
        # 1. Try LangSmith Prompt Hub
        if self.use_langsmith and self.langsmith_client:
            try:
                # We assume a naming convention based on dataset and mode
                sys_prompt_name = f"subgraphrag-sys-{prompt_mode}"
                cot_prompt_name = f"subgraphrag-cot-{prompt_mode}"
                
                sys_prompt = self.langsmith_client.pull_prompt(sys_prompt_name).messages[0].prompt.template
                cot_prompt = self.langsmith_client.pull_prompt(cot_prompt_name).messages[0].prompt.template
                print(f"[PromptManager] Successfully pulled prompts from LangSmith: {sys_prompt_name}, {cot_prompt_name}")
                self.last_fetched_version = f"langsmith-registry:{sys_prompt_name}"
                return sys_prompt, cot_prompt
            except Exception as e:
                print(f"[PromptManager] LangSmith prompt pull failed, falling back. Error: {e}")

        # 2. Fallback to local prompt definitions
        return self.get_local_prompts(prompt_mode, model_name, llm_mode)

    def get_prompts(self, prompt_mode: str, model_name: str, llm_mode: str) -> Tuple[str, str]:
        """
        Attempts to load versioned prompts from registry.
        Uses cached prompts if already resolved.
        """
        cache_key = (prompt_mode, model_name, llm_mode)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        prompts = self._resolve_prompts(prompt_mode, model_name, llm_mode)
        self._cache[cache_key] = prompts
        return prompts
