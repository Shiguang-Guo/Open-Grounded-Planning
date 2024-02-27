"""
@author: Guo Shiguang
@software: PyCharm
@file: embedding.py
@time: 2023/12/4 17:53
"""

from typing import Union

import numpy
from openai import OpenAI

embedding_client = OpenAI(
    api_key="",
    base_url=""
    )


# @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def embedding(sentence: Union[str, list[str]]):
    if isinstance(sentence, str):
        sentence = [sentence]
    response = embedding_client.embeddings.create(
        model="text-embedding-ada-002",
        input=sentence
        )
    return numpy.array([item.embedding for item in response.data])
