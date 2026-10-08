import os
import time
import openai
import requests
from dotenv import load_dotenv
# from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt, custom_sys_prompt

load_dotenv()


def ollama_chat(messages, model="llama3.1", max_tokens=4000, temperature=0, base_url=None):
    if not base_url:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    url = f"{base_url.rstrip('/')}/api/chat"

    clean_model = model.replace("ollama:", "").strip() if model else "llama3.1"
    if clean_model == "llama":
        clean_model = "llama3.1:latest"

    payload = {
        "model": clean_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 4096))
        },
        "keep_alive": -1
    }

    response = requests.post(url, json=payload)
    res_json = response.json()
    if "error" in res_json:
        raise RuntimeError(f"Ollama error: {res_json['error']}")
    return res_json


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, ollama_base_url=None):
    if not ollama_base_url:
        ollama_base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm = None
    if "fpt" in model_name:
        # OpenAI GPT models
        client = OpenAI(api_key=os.getenv("FPT_KEY"), base_url=os.getenv("FPT_BASE_URL"))
        llm = partial(client.chat.completions.create, model="Llama-3.3-70B-Instruct", seed=seed, temperature=temperature, max_tokens=max_tokens)
    elif model_name.startswith("vllm"):
        # vLLM OpenAI-compatible server (e.g. vast.ai vLLM template): -m vllm:<hf model id>, URL from VLLM_URL
        # long generations (max_tokens 4000) under heavy batching can take > 10 min: the client default timeout (600 s)
        # would abort the run, so allow up to 2 h per request
        client = OpenAI(api_key=os.getenv("VLLM_API_KEY", "EMPTY"), base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"),
                        timeout=float(os.getenv("VLLM_TIMEOUT", 7200)), max_retries=2)
        hf_id = model_name.split(":", 1)[1] if ":" in model_name else "meta-llama/Llama-3.1-8B-Instruct"
        llm = partial(client.chat.completions.create, model=hf_id, seed=seed, temperature=temperature, max_tokens=max_tokens,
                      frequency_penalty=frequency_penalty)
    elif "ollama:qwen" in model_name:
        # Ollama models via OpenAI-compatible API
        client = OpenAI(api_key="ollama", base_url=ollama_base_url)
        llm = partial(client.chat.completions.create, model="qwen3.5:9b", temperature=temperature, max_tokens=max_tokens)
    else:
        llm = partial(ollama_chat, model=model_name, max_tokens=max_tokens, temperature=temperature, base_url=ollama_base_url)
    return llm


def get_outputs(outputs, model_name):
    if "ollama:llama" in model_name:
        return outputs["message"]["content"]
    elif "fpt" in model_name or "ollama" in model_name or model_name.startswith("vllm"):
        return outputs.choices[0].message.content
    else:
        return outputs[0].outputs[0].text


def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if "custom" in mode:
        conversation = [
            {"role": "system", "content": prompts['all_query']},
            {"role": "user", "content": prompts['cot_query']}
        ]
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)

def parse_response(response) -> tuple[str, list[str]]:
    if isinstance(response, (list, tuple)):
        candidates = [str(item).strip() for item in response if str(item).strip()]
        response = next(
            (item for item in candidates if "<ans>" in item or "ans:" in item or "<support>" in item),
            candidates[0] if candidates else "",
        )
    else:
        response = "" if response is None else str(response).strip()

    ans = ""
    supports = []

    if "<ans>" in response and "</ans>" in response:
        ans = response.split("<ans>", 1)[1].split("</ans>", 1)[0].strip()
    else:
        ans_lines = [line.strip() for line in response.splitlines() if line.strip().lower().startswith("ans:")]
        if ans_lines:
            ans = "\n".join(ans_lines)

    if "<support>" in response and "</support>" in response:
        support = response.split("<support>", 1)[1].split("</support>", 1)[0].strip()
        supports = [fact.strip() for fact in support.split("\n") if fact.strip()]

    if not ans:
        return "ans: not available", supports

    normalized_ans = []
    for line in ans.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("ans:"):
            normalized_ans.append(line)
    return "\n".join(normalized_ans) if normalized_ans else "ans: not available", supports


def llm_init_langchain(model_name, temperature=0, max_tokens=4000, ollama_base_url="http://localhost:11434/v1"):
    """
    Initializes a LangChain ChatOpenAI model compatible with Ollama and FPT Cloud
    for automatic token usage tracking in LangSmith.
    """
    from langchain_openai import ChatOpenAI
    
    if "fpt" in model_name:
        return ChatOpenAI(
            base_url=os.getenv("FPT_BASE_URL"),
            api_key=os.getenv("FPT_KEY"),
            model="Llama-3.3-70B-Instruct",
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        if "ollama" in model_name:
            model = "llama3.1:latest"
        return ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )


def llm_inf_langchain(llm, prompts, mode):
    """
    LangChain equivalent of llm_inf, utilizing .invoke() to enable LangSmith tracing.
    """
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = llm.invoke(conversation).content
        res.append(outputs)

    if "custom" in mode:
        conversation = [
            {"role": "system", "content": prompts['all_query']},
            {"role": "user", "content": prompts['cot_query']}
        ]
        outputs = llm.invoke(conversation).content
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = llm.invoke(conversation).content
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = llm.invoke(conversation).content
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_all_langchain(llm, each_qa, llm_mode, model_name, max_retries=5):
    """
    LangChain equivalent of llm_inf_all.
    """
    return llm_inf_langchain(llm, each_qa, llm_mode)