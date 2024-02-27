"""
@author: Guo Shiguang
@software: PyCharm
@file: all_in_one_baseline_eval.py
@time: 2023/12/20 13:07
@description: 
"""
import warnings
from typing import Union

warnings.filterwarnings("ignore")
import argparse
import json
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm

import sys

sys.path.append("/mnt/ceph_home/guoshiguang2021/code/ogp")
from generate.dataset_path_dict import dataset_path_dict
from generate.dataset_settings.apibank import APIBankDataset
from generate.dataset_settings.base_dataset import DataItem, TaskDataset
from generate.dataset_settings.gpt4tools import GPT4ToolsDataset
from generate.dataset_settings.saycan import SaycanDataset
from generate.dataset_settings.toolalpaca import ToolAlpacaDataset
from generate.dataset_settings.virtualhome import VirtualHomeDataset
from generate.dataset_settings.wikihow import WikihowDataset
from utils.chat import get_ans
from utils.custom_tools import set_seed
from utils.embedding import embedding


def build_dataset(source_root, dataset_name, model_api) -> TaskDataset:
    dataset_path = os.path.join(source_root, dataset_path_dict[dataset_name])
    actions_path = os.path.join(dataset_path, "actions.txt")
    actions_emb_path = os.path.join(dataset_path, "actions_emb.npy")
    tasks_path = os.path.join(dataset_path, "tasks.json")

    with open(actions_path, 'r') as f:
        actions = [item.strip() for item in f.readlines()]
    actions_emb = np.load(actions_emb_path)

    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    exists_files = os.listdir(predict_path)
    exists_files = [item for item in exists_files if item.endswith(".json")]
    tasks = [item for item in tasks if item["file_name"] not in exists_files]

    if dataset_name.startswith("wikihow"):
        dataset_class = WikihowDataset
    elif dataset_name == "robot-saycan":
        dataset_class = SaycanDataset
    elif dataset_name == "robot-VirtualHome":
        dataset_class = VirtualHomeDataset
    elif dataset_name == "tools-APIBank":
        dataset_class = APIBankDataset
    elif dataset_name == "tools-GPT4Tools":
        dataset_class = GPT4ToolsDataset
    elif dataset_name == "tools-ToolAlpaca":
        dataset_class = ToolAlpacaDataset
    else:
        raise ValueError(f"dataset {dataset_name} not supported")
    return dataset_class(
        task_data=tasks,
        action_lib=actions,
        action_emb=actions_emb,
        model_func=model_api,
        predict_path=predict_path
        )


def return_number_based_on_length(lst):
    length_to_number = {1: 10, 2: 5, 3: 4, 4: 3, 5: 2, 6: 2, 7: 2}
    return length_to_number.get(len(lst), 1)


def process_item(dataset: TaskDataset, task: DataItem, model_api) -> dict:
    record = {
        "dataset": dataset_name,
        "method": method,
        "status": "",
        "data_item": task.__dict__,
        "init_plan": None,
        "retrieved_steps": [],
        "final_ans": None
        }
    if method == "plan_retrieve":
        # generate plan
        init_plan = dataset.get_no_constraints_actions(task)
        if init_plan is None:
            record["status"] = "failed to generate init plan"
            return record
        record["init_plan"] = init_plan
        # retrieve steps
        retrieved_steps = []
        init_plan_emb = embedding([dataset.extract_action(item) for item in init_plan])
        for item in init_plan_emb:
            _, I = dataset.action_emb.search(item.reshape((1, -1)), 2)
            retrieved_steps.extend([dataset.action_lib[i] for i in I[0]])
        # remove duplicates
        seen = set()
        retrieved_steps = [x for x in retrieved_steps if not (x in seen or seen.add(x))]
        record["retrieved_steps"] = retrieved_steps
        if len(retrieved_steps) == 0:
            record["status"] = "failed to retrieve steps"
            return record
        # generate prompt
        retrieved_steps_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(retrieved_steps)])
        init_plan_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(init_plan)])
        prompt = (f"<Task>: {task.title}\n"
                  f"<Method>: {task.method}\n"
                  f"<Initial steps>:\n"
                  f"{init_plan_str}\n\n"
                  f"<Actions in library>:\n"
                  f"{retrieved_steps_str}\n")

        # generate answer
        max_retry_times = 5
        retry_times = 0
        while answer := model_api(query=prompt, temperature=1.0, instruct=dataset.plan_retrieve_instruction,
                                  save_path=f"{predict_path}_chat/{task.file_name}"):
            answer = dataset.check_list_ans(answer)
            retry_times += 1
            if answer is not None:
                if dataset_name.startswith("tools"):
                    api_name_dict = {item.split("DESCRIPTION")[0].strip(): item for item in retrieved_steps}
                    ans_apis = [dataset.extract_action(item) for item in answer]
                    ans_apis = [item.split("DESCRIPTION")[0].strip() for item in ans_apis]
                    if all([item in list(api_name_dict.keys()) + retrieved_steps for item in ans_apis]):
                        answer = [api_name_dict.get(item, item) for item in ans_apis]
                        break
                else:
                    if all([dataset.extract_action(item.strip()) in retrieved_steps for item in answer]):
                        break
            if retry_times > max_retry_times:
                record["status"] = "failed to generate answer"
                return record

        record["final_ans"] = answer
        record["status"] = "success"
    elif method == "task_retrieve":
        # retrieve steps
        retrieved_steps = []
        item_emd = embedding(task.title)
        _, I = dataset.action_emb.search(item_emd.reshape((1, -1)), 20)
        retrieved_steps.extend([dataset.action_lib[i] for i in I[0]])
        # remove duplicates
        seen = set()
        retrieved_steps = [x for x in retrieved_steps if not (x in seen or seen.add(x))]
        record["retrieved_steps"] = retrieved_steps
        if len(retrieved_steps) == 0:
            record["status"] = "failed to retrieve steps"
            return record
        # generate prompt
        retrieved_steps_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(retrieved_steps)])
        prompt = (f"<Task>: {task.title}\n"
                  f"<Method>: {task.method}\n"
                  f"<Actions in library>:\n"
                  f"{retrieved_steps_str}\n")

        # generate answer
        max_retry_times = 5
        retry_times = 0
        while answer := model_api(query=prompt, temperature=1.0, instruct=dataset.task_retrieve_instruction,
                                  save_path=f"{predict_path}_chat/{task.file_name}"):
            answer = dataset.check_list_ans(answer)
            retry_times += 1
            if answer is not None:
                # for tools
                if dataset_name.startswith("tools"):
                    api_name_dict = {item.split("DESCRIPTION")[0].strip(): item for item in retrieved_steps}
                    ans_apis = [dataset.extract_action(item) for item in answer]
                    ans_apis = [item.split("DESCRIPTION")[0].strip() for item in ans_apis]
                    if all([item in list(api_name_dict.keys()) + retrieved_steps for item in ans_apis]):
                        answer = [api_name_dict.get(item, item) for item in ans_apis]
                        break
                else:
                    if all([dataset.extract_action(item.strip()) in retrieved_steps for item in answer]):
                        break
            if retry_times > max_retry_times:
                record["status"] = "failed to generate answer"
                return record
        record["final_ans"] = answer
        print(answer)
    elif method == "select":
        max_select_loops = 20
        select_loops = 0
        current_plan = []
        record["select_iterations"] = []
        while True:
            select_iterations = {
                "select_loops": select_loops,
                "current_plan": current_plan.copy(),
                "generated_step": None,
                "retrieved_steps": [],
                "chosen_step": None,
                }
            if len(current_plan) == 0:
                current_plan_str = "None"
            else:
                current_plan_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)])
            gene_prompt = (f"<Task>: {task.title}\n"
                           f"<Method>: {task.method}\n"
                           f"<Current plan>:\n"
                           f"{current_plan_str}")
            max_retry_times = 5
            retry_times = 0
            instr = dataset.select_initial_step_instruction if select_loops == 0 else dataset.select_next_step_instruction
            while gene_ans := model_api(query=gene_prompt, temperature=1.0,
                                        instruct=instr, save_path=f"{predict_path}_chat/{task.file_name}"):
                if "[New step: None]" in gene_ans and select_loops != 0:
                    break
                gene_ans = dataset.check_select_ans(gene_ans)
                if gene_ans is not None:
                    break
                retry_times += 1
                if retry_times > max_retry_times:
                    record["status"] = "failed to generate answer"
                    return record
            gene_ans = dataset.extract_action(gene_ans)
            select_iterations["generated_step"] = gene_ans
            if "[New step: None]" in gene_ans:
                record["status"] = "finish generate"
                select_iterations["generated_step"] = gene_ans
                record["select_iterations"].append(select_iterations)
                break
            item_emd = embedding(gene_ans)
            _, I = dataset.action_emb.search(item_emd.reshape((1, -1)), 5)
            retrieved_steps = [dataset.action_lib[i] for i in I[0]]
            select_iterations["retrieved_steps"] = retrieved_steps.copy()
            retrieved_steps_str = "\n".join([f"{item}" for idx, item in enumerate(retrieved_steps)])
            select_prompt = (f"<Task>: {task.title}\n"
                             f"<Method>: {task.method}\n"
                             f"<Current plan>:\n"
                             f"{current_plan_str}\n"
                             f"<Actions in library>:\n"
                             f"{retrieved_steps_str}\n")
            retry_times = 0
            while select_ans := model_api(query=select_prompt, temperature=1.0,
                                          instruct=dataset.select_select_instruction,
                                          save_path=f"{predict_path}_chat/{task.file_name}"):
                if "None of these" in select_ans or select_ans == "[New step: None]":
                    break
                select_ans = dataset.check_select_ans(select_ans)
                if select_ans is not None:
                    select_ans = dataset.extract_action(select_ans)
                    if dataset_name.startswith("tools"):
                        api_name_dict = {item.split("DESCRIPTION")[0].strip(): item for item in retrieved_steps}
                        select_ans = select_ans.split("DESCRIPTION")[0].strip()
                        if select_ans in retrieved_steps + list(api_name_dict.keys()):
                            select_ans = api_name_dict.get(select_ans, select_ans)
                            break
                    else:
                        if select_ans in retrieved_steps:
                            break
                retry_times += 1
                if retry_times > max_retry_times:
                    record["status"] = "failed to select answer"
                    return record

            if "None of these" in select_ans or select_ans == "[New step: None]":
                record["status"] = "reject all candidates"

            select_iterations["chosen_step"] = select_ans
            current_plan.append(select_ans)
            record["select_iterations"].append(select_iterations)

            record["retrieved_steps"].extend(retrieved_steps.copy())
            record["final_ans"] = [f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)]

            select_loops += 1
            if select_loops >= max_select_loops:
                record["status"] = "select loops exceed max_select_loops"
                return record
    elif method == "dfs":
        def next_node(current_plan, task, is_initial=False):
            if len(current_plan) == 0:
                current_plan_str = "None"
            else:
                current_plan_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)])
            gene_prompt = (f"<Task>: {task.title}\n"
                           f"<Method>: {task.method}\n"
                           f"<Current plan>:\n"
                           f"{current_plan_str}")

            max_retry_times = 5
            retry_times = 0
            status = "success"
            instr = dataset.dfs_initial_step_instruction if is_initial else dataset.dfs_next_step_instruction
            while gene_ans := model_api(query=gene_prompt, temperature=1.0, instruct=instr,
                                        save_path=f"{predict_path}_chat/{task.file_name}"):
                if "[None]" in gene_ans and not is_initial:
                    return "success", [{"cand": "[None]", "current_plan": current_plan.copy()}]
                gene_ans = dataset.check_dfs_candidates_ans(gene_ans)
                if gene_ans is not None:
                    break
                retry_times += 1
                if retry_times > max_retry_times:
                    status = "failed to generate answer"
                    break

            if status == "success":
                gene_ans = [{"cand": dataset.extract_action(item), "current_plan": current_plan.copy()}
                            for item in gene_ans]
            return status, gene_ans[::-1]

        record["dfs_iterations"] = []
        max_dfs_loops = 30
        dfs_loops = 0
        gene_status, node = next_node([], task, is_initial=True)
        if gene_status != "success":
            record["status"] = gene_status
            return record
        else:
            current_plan_stack = node
        while True:
            dfs_loops += 1
            dfs_iterations = {
                "next_operation": "Finish",
                "dfs_loops": dfs_loops,
                "current_plan_stack": current_plan_stack.copy(),
                }
            if not current_plan_stack:
                gene_status, node = next_node([], task, is_initial=True)
                if gene_status != "success":
                    record["status"] = gene_status
                    return record
                else:
                    current_plan_stack = node
                    dfs_iterations["new_current_plan_stack"] = current_plan_stack.copy()

            current_node = current_plan_stack.pop()
            dfs_iterations["current_node"] = current_node.copy()

            if dfs_loops > max_dfs_loops:
                record["status"] = "dfs loops exceed max_dfs_loops"
                return record

            next_step = current_node["cand"]
            current_plan = current_node["current_plan"].copy()

            if "[None]" in current_node["cand"]:
                dfs_iterations["next_operation"] = "finish"
                record["dfs_iterations"].append(dfs_iterations)
                record["status"] = "success"
                return record

            if len(current_plan) == 0:
                current_plan_str = "None"
            else:
                current_plan_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)])
            item_emd = embedding(next_step)
            _, I = dataset.action_emb.search(item_emd.reshape((1, -1)), 5)
            retrieved_steps = [dataset.action_lib[i] for i in I[0]]
            dfs_iterations["retrieved_steps"] = retrieved_steps.copy()
            retrieved_steps_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(retrieved_steps)])
            select_prompt = (f"<Task>: {task.title}\n"
                             f"<Method>: {task.method}\n"
                             f"<Current plan>:\n"
                             f"{current_plan_str}\n"
                             f"<Actions in library>:\n"
                             f"{retrieved_steps_str}\n")
            max_retry_times = 5
            retry_times = 0
            while select_ans := model_api(query=select_prompt, temperature=1.0,
                                          instruct=dataset.dfs_select_instruction,
                                          save_path=f"{predict_path}_chat/{task.file_name}"):
                if "None of these" in select_ans or select_ans == "[New step: None]":
                    break
                select_ans = dataset.check_select_ans(select_ans)
                if select_ans is not None:
                    select_ans = dataset.extract_action(select_ans)
                    if dataset_name.startswith("tools"):
                        api_name_dict = {item.split("DESCRIPTION")[0].strip(): item for item in retrieved_steps}
                        select_ans = select_ans.split("DESCRIPTION")[0].strip()
                        if select_ans in retrieved_steps + list(api_name_dict.keys()):
                            select_ans = api_name_dict.get(select_ans, select_ans)
                            break
                    else:
                        if select_ans in retrieved_steps:
                            break
                retry_times += 1
                if retry_times > max_retry_times:
                    dfs_iterations["chosen_step"] = "failed to select answer"
                    record["dfs_iterations"].append(dfs_iterations)
                    dfs_iterations["next_operation"] = "rollback"
                    break

            if dfs_iterations["next_operation"] == "rollback":
                continue

            dfs_iterations["chosen_step"] = select_ans
            if "None of these" in select_ans or select_ans == "[New step: None]":
                dfs_iterations["next_operation"] = "rollback"
            else:
                current_plan.append(select_ans)
                gene_status, node = next_node(current_plan, task)
                if gene_status == "success":
                    current_plan_stack.extend(node.copy())
                    dfs_iterations["next_operation"] = "expand"
                    dfs_iterations["next_node"] = node.copy()
                else:
                    dfs_iterations["next_operation"] = "rollback"
            record["dfs_iterations"].append(dfs_iterations)
            record["final_ans"] = [f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)]
    elif method == "rewrite":
        reference_steps = retrieve_and_rm_dupli(task.title, dataset)
        record["reference_steps"] = reference_steps

        # initial plan
        reference_steps_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(reference_steps)])
        prompt = (f"<Task>: {task.title}\n"
                  f"<Method>: {task.method}\n"
                  f"<References>:\n"
                  f"{reference_steps_str}\n")
        max_retry_times = 5
        retry_times = 0
        while init_plan := model_api(query=prompt, temperature=1.0, instruct=dataset.refine_initial_plan,
                                     save_path=f"{predict_path}_chat/{task.file_name}"):
            init_plan = dataset.check_list_ans(init_plan)
            retry_times += 1
            if init_plan is not None:
                break
            if retry_times > max_retry_times:
                record["status"] = "failed to generate initial plan"
                return record
        record["init_plan"] = init_plan

        record["rewrite"] = []
        # rewrite
        max_rewrite_loops = 20
        rewrite_loops = 0
        current_plan = []
        for item in init_plan:
            if item in reference_steps:
                current_plan.append(f"{{<IN LIB> {dataset.extract_action(item)}}}")
            else:
                current_plan.append(f"{{<TO BE REPLACED> {dataset.extract_action(item)}}}")
        while rewrite_loops < max_rewrite_loops:
            rewrite = {
                "rewrite_loops": rewrite_loops,
                "current_plan": current_plan.copy(),
                "retrieved_steps": [],
                "new_plan": None,
                }
            # if rewrite_loops >= 2:
            #     rewritten_last_one_num = [item for item in record["rewrite"][-1]["new_plan"] if
            #                               item.startswith("{<IN LIB>")]
            #     rewritten_last_two_num = [item for item in record["rewrite"][-2]["new_plan"] if
            #                               item.startswith("{<IN LIB>")]
            #     if rewritten_last_one_num == rewritten_last_two_num:
            #         past_retrieved_steps = (record["rewrite"][-1]["retrieved_steps"]
            #                                 + record["rewrite"][-2]["retrieved_steps"])
            #         feedback_steps = dataset.feedback(task, past_retrieved_steps, current_plan)
            #         if feedback_steps is not None:
            #             current_plan = feedback_steps
            #             rewrite["feedback"] = {
            #                 "past_retrieved_steps": past_retrieved_steps,
            #                 "current_plan": current_plan,
            #                 "feedback": feedback_steps
            #                 }

            not_in_lib_steps = [step for step in current_plan if step.startswith("{<TO BE REPLACED>")][:3]
            retrieved_steps = []
            for emb in embedding([dataset.extract_action(step) for step in not_in_lib_steps]):
                _, I = dataset.action_emb.search(emb.reshape((1, -1)), 100)
                raw_retrieved_steps = [dataset.action_lib[i] for i in I[0]][
                                      :return_number_based_on_length(not_in_lib_steps)]
                retrieved_steps.extend(raw_retrieved_steps)
            if len(retrieved_steps) == 0:
                record["status"] = "failed to retrieve steps"
                return record
            seen = set()
            retrieved_steps = [x for x in retrieved_steps if not (x.lower() in seen or seen.add(x.lower()))]
            rewrite["retrieved_steps"] = retrieved_steps.copy()

            current_plan_str = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(current_plan)])
            retrieved_steps_str = "\n".join([f"{{<IN LIB> {item}}}" for item in retrieved_steps])

            # rewrite
            prompt = dataset.refined_2_shot_examples.copy()
            prompt.append(f"<Task>: {task.title}\n"
                          f"<Method>: {task.method}\n"
                          f"<Current plan>:\n"
                          f"{current_plan_str}\n\n"
                          f"<Actions in library>:\n"
                          f"{retrieved_steps_str}")

            max_retry_times = 5
            retry_times = 0
            in_library_steps = [dataset.extract_action(item) for item in current_plan if item.startswith("{<IN LIB>")]
            in_library_steps.extend(retrieved_steps)
            while rewrite_ans := model_api(query=prompt, temperature=1.0, instruct=dataset.refine_instruction,
                                           save_path=f"{predict_path}_chat/{task.file_name}"):
                rewrite_ans = dataset.check_refine_ans(rewrite_ans)
                retry_times += 1
                if rewrite_ans is not None:
                    rewrite_ans = [dataset.extract_action(item) for item in rewrite_ans]
                    rewrite_ans = [
                        f"{{<IN LIB> {item}}}" if item in in_library_steps else f"{{<TO BE REPLACED> {item}}}" for item
                        in rewrite_ans]
                    if any([dataset.extract_action(item) in in_library_steps for item in rewrite_ans if
                            item.startswith("{<IN LIB>")]):
                        break
                if retry_times > max_retry_times:
                    record["status"] = "failed to rewrite"
                    return record

            current_plan = rewrite_ans.copy()
            rewrite["new_plan"] = current_plan.copy()
            record["final_ans"] = [f"{idx + 1}. {dataset.extract_action(item)}" for idx, item in
                                   enumerate(current_plan)]
            rewrite_loops += 1
            record["rewrite"].append(rewrite)
            if all([item.startswith("{<IN LIB>") for item in rewrite_ans]):
                break
        if rewrite_loops >= max_rewrite_loops:
            record["status"] = "rewrite loops exceed max_rewrite_loops"
            return record

    else:
        raise ValueError(f"method {method} not supported")

    return record


def retrieve_and_rm_dupli(emb_str: Union[str, list[str]], dataset: TaskDataset):
    retrieved_steps = []
    item_emd = embedding(emb_str)
    _, I = dataset.action_emb.search(item_emd.reshape((1, -1)), 20)
    retrieved_steps.extend([dataset.action_lib[i] for i in I[0]])
    # remove duplicates
    seen = set()
    retrieved_steps = [x for x in retrieved_steps if not (x in seen or seen.add(x))]
    return retrieved_steps


def process_item_wrapper(item) -> dict:
    try:
        record = process_item(dataset, item, model_api)
        with open(os.path.join(predict_path, item.file_name), 'w') as f:
            f.write(json.dumps(record, indent=4, ensure_ascii=False))
        return record
    except Exception as e:
        print(e)
        return {"error": str(e), "data_item": item.__dict__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatgpt")
    parser.add_argument("--dataset", type=str, default="wikihow-Computers and Electronics")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--method", type=str, default="plan_retrieve")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()
    set_seed(42)
    source_path = "/mnt/ceph_home/guoshiguang2021/code/opendomain/datasets/ogp"
    predict_path = "/mnt/ceph_home/guoshiguang2021/code/ogp/predict"
    dataset_name = args.dataset
    version = args.version
    method = args.method  # chosen from ["plan_retrieve", "task_retrieve", "select", "dfs", "rewrite"]

    if version != "":
        version = "_" + version

    predict_path = os.path.join(predict_path, args.model, method + version, dataset_name)

    os.makedirs(predict_path, exist_ok=True)
    os.makedirs(predict_path + "_chat", exist_ok=True)

    model_api = partial(get_ans, model=args.model, workers=args.max_workers)
    # build dataset
    dataset = build_dataset(source_path, dataset_name, model_api)

    progress_bar = tqdm(total=len(dataset.task_data), desc=f"generating {args.dataset} {args.method}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item_wrapper, item) for item in dataset.task_data]

        for future in as_completed(futures):
            results.append(future.result())
            progress_bar.update(1)

    progress_bar.close()

    # Append results to all_records while maintaining order
    errors = [item for item in results if "error" in item]

    if len(errors) > 0:
        print(f"Dataset {args.dataset} still remain {len(errors)} errors.")
        with open(os.path.join(predict_path, "error.json"), 'w') as f:
            json.dump(errors, f, indent=4)
    else:
        if os.path.exists(os.path.join(predict_path, "error.json")):
            os.remove(os.path.join(predict_path, "error.json"))
