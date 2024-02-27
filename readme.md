# Open Grounded Planning: Challenges and Benchmark Construction
<p>
📃 <a href="">ArXiv Paper</a>
</p>

## Introduction
The emergence of large language models (LLMs) has increasingly drawn attention to the use of LLMs for human-like planning. Existing work on LLM-based planning either focuses on leveraging the inherent language generation capabilities of LLMs to produce free-style plans or employs reinforcement learning approaches to learn decision-making for a limited set of actions within restricted environments. However, both approaches exhibit significant discrepancies between the open and executable requirements in real-world planning. In this paper, we propose a new planning task---open grounded planning. The primary objective of open grounded planning is to ask the model to generate an executable plan based on a variable action set, thereby ensuring the executability of the produced plan. To this end, we establish a benchmark for open grounded planning spanning a wide range of domains. Then we test current state-of-the-art LLMs along with five planning approaches, revealing that existing LLMs and methods still struggle to address the challenges posed by grounded planning in open domains. The outcomes of this paper define and establish a foundational dataset for open grounded planning, and shed light on the potential challenges and future directions of LLM-based planning.

## Benchmark
The benchmark we built is in the `datasets` directory, which contains `xxx_full` and `xxx_500`. We sample up to 500 in each category for evaluation and provide the complete dataset for further study. Each dataset contains `tasks.json` and `actions.txt`:
* `tasks.json`: Tasks are stored in json format, including task names, methods and steps. Note: There are no specified methods in robots and tools, we filled in the default values to make it more consistent with the format of the dataset.
* `actions.txt`: Each line represents an action. The action sets of full set and evaluation set are the same.

## Code
We provide experimental code for the various methods described in the paper under `generate/generate_loop.py`. You may need to modify the path and add the openai key under `utils/chat.py` and `utils/embedding.py` to run the script correctly.
We used `text-embedding-ada-002` to generate embeddings for each action. We did not upload the generated embeddings due to the large size of the generated files. You can use `utils/embedding.py` to easily generate an embedding for each action in the action sets, then save all the embeddings to a file and place it in the corresponding dataset directory to run the experiment.


