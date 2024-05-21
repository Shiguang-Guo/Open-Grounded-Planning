import json
import os
import sys
import re
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

sys.path.append("..")

from eval_utils import Evaluator


if __name__ == "__main__":
    eval_records_dir = ""                   # place the path of your models' generation results here
    eval_result_dir = "./eval_results/"     # path of the evaluation results saved

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', type=str, default="odp")                    # models to be evaluated, including: [sft, chatgpt, vicuna, llama]
    parser.add_argument('--version', type=str, default="demo")                      # version of evaluation
    parser.add_argument('--baseline_method', type=str, default="plan_retrieve")     # type of baseline method, including: [plan_retrieve, task_retrieve, select, dfs, rewrite]
    parser.add_argument('--eval_set_type', type=str, default="wikihow")             # type of evaluation set, including: [wikihow, tools, robot]
    parser.add_argument('--eval_fast', type=str, default="False")                   # if do fast evaluation, only 100 cases will be tested, otherwise the whole eval set will be under evaluation

    args = parser.parse_args()

    eval_model = ""
    # parsing target model to be evaluated
    eval_model = args.eval_model

    # parsing evaluation version
    eval_version = args.version

    # parsing baseline_method
    baseline_method = args.baseline_method

    # parsing eval_set
    eval_set_type = args.eval_set_type

    # parsing fast evaluation flag
    eval_fast = args.eval_fast

    evaluator = Evaluator(eval_model=eval_model,
                        eval_version=eval_version, 
                        baseline_method=baseline_method,
                        eval_set_type=eval_set_type,
                        eval_records_dir=eval_records_dir,
                        eval_result_dir=eval_result_dir,
                        eval_fast=eval_fast)
    evaluator.eval()

    