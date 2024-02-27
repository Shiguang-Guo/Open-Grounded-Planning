"""
@author: Guo Shiguang
@software: PyCharm
@file: chat.py
@time: 2023/9/1 17:45
"""
import json
from datetime import datetime
from typing import Union

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM_PROMPT = f"""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
Knowledge cutoff: 2021-09
Current date: {datetime.now().date()}"""

local_client_list = None

vicuna_client = OpenAI(
    base_url=f"",
    api_key=""
    )
sft_client = OpenAI(
    base_url="",
    api_key=""
    )

chatgpt_client = OpenAI(
    base_url="",
    api_key=""
    )


def get_vicuna(query, temperature=0, instruct=None):
    if isinstance(query, str):
        query = [query]
    if instruct is not None:
        query[0] = instruct + "\n\n" + query[0]
    query_ = []
    for idx, item in enumerate(query):
        if idx % 2 == 0:
            query_.append({"role": "user", "content": item})
        else:
            query_.append({"role": "assistant", "content": item})

    model = "vicuna-7b-16k"

    completion = vicuna_client.chat.completions.create(
        model=model, messages=query_, temperature=temperature
        )
    return completion.choices[0].message.content


def get_sft(query, temperature=0, instruct=None):
    if isinstance(query, str):
        query = [query]
    if instruct is not None:
        query[0] = instruct + "\n\n" + query[0]
    query_ = []
    for idx, item in enumerate(query):
        if idx % 2 == 0:
            query_.append({"role": "user", "content": item})
        else:
            query_.append({"role": "assistant", "content": item})

    model = "sft"

    completion = sft_client.chat.completions.create(
        model=model, messages=query_, temperature=temperature
        )
    return completion.choices[0].message.content


def get_chatgpt(query, model, temperature=0, instruct=None):
    if isinstance(query, str):
        query = [query]
    query_ = []
    if instruct is None:
        query_.append({"role": "system", "content": SYSTEM_PROMPT})
    else:
        query_.append({"role": "system", "content": instruct})

    for idx, item in enumerate(query):
        if idx % 2 == 0:
            query_.append({"role": "user", "content": item})
        else:
            query_.append({"role": "assistant", "content": item})

    model = "gpt-3.5-turbo-1106"

    completion = chatgpt_client.chat.completions.create(
        model=model, messages=query_, temperature=temperature
        )
    return completion.choices[0].message.content

def get_ans(query: Union[str, list[str]], instruct: str = None, model: str = "1106", temperature=0, top_p=1,
            save_path=None
            ) -> str:
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
    def inner_chat_retry(query: Union[str, list[str]], instruct: str = None, model: str = "1106", temperature=0,
                         top_p=1, ):
        if model == "vicuna":
            return get_vicuna(query, temperature=temperature, instruct=instruct)
        if model == "sft":
            return get_sft(query, temperature=temperature, instruct=instruct)
        elif "chatgpt" in model:
            return get_chatgpt(query, model, temperature=temperature, instruct=instruct)
        else:
            raise ValueError("model name error")

    result = inner_chat_retry(query, instruct, model, temperature, top_p, )
    if save_path is not None:
        with open(save_path, "a") as f:
            instruct_ = {
                "query": query, "response": result, "model": model, "temperature": temperature, "top_p": top_p,
                "instruct": instruct
                }
            json.dump(instruct_, f, ensure_ascii=False)
            f.write("\n")
    return result
