import os
import time
import openai
import requests
from dotenv import load_dotenv
# from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt

load_dotenv()


def ollama_chat(messages, model="llama3.1", max_tokens=4000, temperature=0):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "llama3.1",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "keep_alive": "10m"
    }

    response = requests.post(url, json=payload)
    return response.json()


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, ollama_base_url="http://0.0.0.0:8888/v1"):
    llm = None
    if "gpt" in model_name:
        # OpenAI GPT models
        client = OpenAI(api_key=os.getenv("FPT_KEY"), base_url=os.getenv("FPT_BASE_URL"))
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    elif "ollama:qwen" in model_name:
        # Ollama models via OpenAI-compatible API
        client = OpenAI(api_key="ollama", base_url=ollama_base_url)
        llm = partial(client.chat.completions.create, model="qwen3.5:9b", temperature=temperature, max_tokens=max_tokens)
    else:
        llm = partial(ollama_chat, model=model_name, max_tokens=max_tokens, temperature=temperature)
    return llm


def get_outputs(outputs, model_name):
    if "ollama:llama" in model_name:
        return outputs["message"]["content"]
    elif "gpt" in model_name or "ollama" in model_name:
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
