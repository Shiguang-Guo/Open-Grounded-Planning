import json
import os
import sys
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures


sys.path.append("..")

from eval_prompts import eval_compare_plan_prompt
from utils.chat import get_ans



score_flag = True
extract_flag = True

def create_dir(folder_dir):
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            print("Failed to creating dir '{folder_dir}': {e}")
    else:
        print("Dir exists! ", folder_dir)


class Evaluator:
    def __init__(self, eval_model, eval_version, baseline_method, eval_set_type, eval_records_dir, eval_result_dir, eval_fast):
        """"""
        self.eval_model = eval_model
        self.eval_version = eval_version
        self.baseline_method = baseline_method
        self.eval_set_type = eval_set_type
        self.eval_records_dir = eval_records_dir
        self.eval_result_base_dir = eval_result_dir
        self.eval_result_dir = ""
        self.eval_fast = eval_fast


    def eval(self):
        """"""
        if self.eval_set_type == "wikihow":
            self.eval_result_dir = os.path.join(self.eval_result_base_dir, "wikihow_eval")
        elif self.eval_set_type == "tools":
            self.eval_result_dir = os.path.join(self.eval_result_base_dir, "tools_eval")
        elif self.eval_set_type == "robot":
            self.eval_result_dir = os.path.join(self.eval_result_base_dir, "robot_eval")
        create_dir(self.eval_result_dir)

        self.eval_baseline()


    def eval_baseline(self):
        """"""
        predicted_data_dir = os.path.join(self.eval_records_dir, self.eval_model, self.baseline_method)
        
        eval_set_size_dict = json.load(open("./eval_set_size.json", "r"))

        temp_baseline_eval_result_dir = self.eval_model + "_" + self.eval_version + "_" + self.baseline_method
        baseline_eval_result_dir = os.path.join(self.eval_result_dir, temp_baseline_eval_result_dir)
        create_dir(baseline_eval_result_dir)
        category_list = []
        if self.eval_set_type == "wikihow":
            category_list = [category for category in os.listdir(predicted_data_dir) if "wikihow" in category and "chat" not in category]    
        elif self.eval_set_type == "tools":
            category_list = [category for category in os.listdir(predicted_data_dir) if "tools" in category and "chat" not in category]
        elif self.eval_set_type == "robot":
            category_list = [category for category in os.listdir(predicted_data_dir) if "robot" in category and "chat" not in category]
        print(category_list)

        for category_name in category_list:
            print(category_name)
            category_dir = os.path.join(predicted_data_dir, category_name)
            eval_data = self.read_infer_baseline(category_dir, category_name)
            case_count = eval_set_size_dict[category_name]
            print("eval_data_length:", case_count)

            formated_eval_data = []
            for item in eval_data:
                if "status" in item.keys():
                    if item["status"] == "failed to rewrite":
                        print("failed to rewrite")
                        continue
                
                if item["final_ans"] == [] or item["final_ans"] == None:
                    print("Final Answer not Produced!")
                else:
                    formated_eval_data.append(self.format_answer(item))
            valid_case_count = len(formated_eval_data)

            real_valid_eval_data = []
            hallu_count_total = 0
            for eval_data_dict in formated_eval_data:
                hallu_count = self.count_hallu_baseline(eval_data_dict)
                hallu_count_total += hallu_count
                if hallu_count == 0:
                    real_valid_eval_data.append(eval_data_dict)

            success_count = len(real_valid_eval_data)
            success_rate = round(success_count / case_count, 2)
            avg_hallu_count = round(hallu_count_total / valid_case_count, 2) if valid_case_count > 0 else 0

            while not os.path.exists(os.path.join(baseline_eval_result_dir, category_name, "final_score.json")):
                print(os.path.join(baseline_eval_result_dir, category_name, "final_score.json"))
                if score_flag:
                    self.score_tool(baseline_eval_result_dir, category_name, real_valid_eval_data)

                if extract_flag:
                    win_count = self.extract_win_count(baseline_eval_result_dir, category_name)
                    result_category_dir = os.path.join(baseline_eval_result_dir, category_name)
                    score_result_path = os.path.join(result_category_dir, "final_score.json")
                    
                    win_rate = round(win_count / len(real_valid_eval_data), 2) if len(real_valid_eval_data) > 0 else 0
                    final_score_dict = {
                        "category_name": category_name,
                        "case_count": case_count,
                        "not_none_case_count": valid_case_count,
                        "valid_case_count": len(real_valid_eval_data),
                        "success_rate": success_rate,
                        "total_hallu_count": hallu_count_total,
                        "avg_hallu_count": avg_hallu_count,
                        "win_count": win_count,
                        "win_rate": win_rate
                    }
                    print("win rate: ", win_rate)
                    with open(score_result_path, "w") as final_score_file:
                        json.dump(final_score_dict, final_score_file)


    def score_tool(self, eval_result_dir, category_name, eval_data):
        category_dir = os.path.join(eval_result_dir, category_name)
        create_dir(category_dir)
        if os.path.exists(os.path.join(category_dir, "final_score.json")):
            return
        temp_score_path = os.path.join(category_dir, "score_temp.json")
        temp_score_path_reverse = os.path.join(category_dir, "score_temp_reverse.json")

        print("Scoring Loop 1:")
        with open(temp_score_path, "w") as temp_score_file:
            temp_score_list = []
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(lambda x: self.score_tool_wrapper(x, False), eval_data), total=len(eval_data)))
                temp_score_list.extend(results)

            json.dump(temp_score_list, temp_score_file)
        
        print("Scoring Loop 2:")
        with open(temp_score_path_reverse, "w") as temp_score_file:
            temp_score_list = []
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(lambda x: self.score_tool_wrapper(x, True), eval_data), total=len(eval_data)))
                temp_score_list.extend(results)

            json.dump(temp_score_list, temp_score_file)


    def score_tool_wrapper(self, item, reverse_flag):
        file_name = item["file_name"]
        task = item["title"]
        method = item["method"]

        standard_answer = "\n".join(step.split("DESCRIPTION")[0] for step in item["golden_answer"])
        model_answer = "\n".join(step for step in item["final_answer"])


        score_result = ""
        while score_result == "":
            score_result = self.auto_score_kit(task=task, method=method, standard_answer=standard_answer, model_answer=model_answer, reverse_flag=reverse_flag)
            
            temp_ret_dict = {
                "file_name": file_name,
                "title": task,
                "method": method,
                "score_str": score_result
            }
            if score_result == "":
                print("Scoring Failed! Retrying!")
        
        return temp_ret_dict


    def auto_score_kit(self, task, method, standard_answer, model_answer, reverse_flag=False):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                if reverse_flag:
                    score_str = get_ans(eval_compare_plan_prompt.format(TASK=task, METHOD=method, PLAN1=model_answer, PLAN2=standard_answer), model="1106", temperature=0.0)
                else:
                    score_str = get_ans(eval_compare_plan_prompt.format(TASK=task, METHOD=method, PLAN1=standard_answer, PLAN2=model_answer), model="1106", temperature=0.0)

                return score_str

            except Exception as e:
                print(f"Caught an Exception: {e}")
                retries += 1
        return ""


    def extract_win_count(self, eval_result_dir, category_name):
        """"""
        category_dir = os.path.join(eval_result_dir, category_name)
        temp_score_path = os.path.join(category_dir, "score_temp.json")
        temp_score_path_reverse = os.path.join(category_dir, "score_temp_reverse.json")

        temp_win_count = 0
        
        with open(temp_score_path, "r") as temp_score_file_in:
            score_data = json.load(temp_score_file_in)
            for score_line in score_data:
                temp_win_count += self.extract_win(score_line["score_str"], reverse_flag=False)  

        with open(temp_score_path_reverse, "r") as temp_score_file_in:
            score_data = json.load(temp_score_file_in)
            for score_line in score_data:
                temp_win_count += self.extract_win(score_line["score_str"], reverse_flag=True)  
        
        return temp_win_count / 2


    def extract_win(self, score_str, reverse_flag):
        """"""
        pattern_better_plan = r'\<Better Plan\>[\s]*(.*?)[\s]*\</Better Plan\>'

        better_plan = re.findall(pattern_better_plan, score_str)

        if reverse_flag:
            if "Plan1" in better_plan or "Plan 1" in better_plan or "Both" in better_plan:
                return 1
            else:
                return 0
        else:
            if "Plan2" in better_plan or "Plan 2" in better_plan or "Both" in better_plan:
                return 1
            else:
                return 0
            

    def count_hallu_baseline(self, eval_data_dict):
        hallu_count = 0
        answer_steps = eval_data_dict["final_answer"]
        retrieved_steps_str = "\n".join(eval_data_dict["retrieved_steps"])
        for answer_action in answer_steps:
            if answer_action not in retrieved_steps_str:
                if "TO BE REPLACED" not in answer_action:
                    hallu_count += 1
        
        return hallu_count

    
    def read_infer_baseline(self, eval_file_path, category_name,):
        ori_file_name_list = os.listdir(eval_file_path)
        file_name_list = []
        for file_name in ori_file_name_list:
            if file_name.endswith(".json"):
                if file_name.split(".")[0] != (category_name + "_v0") and "error" not in file_name:
                    file_name_list.append(file_name)
        print(len(file_name_list))
        if self.eval_fast == "True":
            file_name_list = file_name_list[:100] if len(file_name_list) > 100 else file_name_list
        else:
            file_name_list = file_name_list[:500] if len(file_name_list) > 500 else file_name_list
        eval_data_list = []
        for file_name in file_name_list:
            file_path = os.path.join(eval_file_path, file_name)
            with open(file_path, "r") as data_file:
                eval_data_list.append(json.load(data_file))
        
        return eval_data_list


    def format_answer(self, data_item):
        """"""
        ret_dict = {
            "file_name": "",
            "title": "",
            "method": "",
            "golden_answer": [],
            "retrieved_steps": [],
            "final_answer": []
        }

        ret_dict["file_name"] = data_item["data_item"]["file_name"]
        ret_dict["title"] = data_item["data_item"]["title"]
        ret_dict["method"] = data_item["data_item"]["method"]
        ret_dict["golden_answer"] = data_item["data_item"]["steps"]
        retrieved_steps = data_item["retrieved_steps"]


        # formatting "retrieved_steps" for methods "rewrite", "select" and "dfs"
        if "rewrite" in self.baseline_method:
            retrieved_steps = []
            for selection_loop in data_item["rewrite"]:
                for selected_step in selection_loop["retrieved_steps"]:
                    if selected_step not in retrieved_steps:
                        retrieved_steps.append(selected_step)

        elif "select" in self.baseline_method:
            retrieved_steps = []
            for selection_loop in data_item["select_iterations"]:
                for selected_step in selection_loop["retrieved_steps"]:
                    if selected_step not in retrieved_steps:
                        retrieved_steps.append(selected_step)

        elif "dfs" in self.baseline_method:
            retrieved_steps = []
            for selection_loop in data_item["dfs_iterations"]:
                if "retrieved_steps" in selection_loop.keys():
                    for selected_step in selection_loop["retrieved_steps"]:
                        if selected_step not in retrieved_steps:
                            retrieved_steps.append(selected_step)
        
        ret_dict["retrieved_steps"] = retrieved_steps
            
        for step in data_item["final_ans"]:
            if step != "":
                if step.split(".")[0].isdigit():
                    step = step.replace(step.split(".")[0] + ".", "").strip()
                if "DESCRIPTION" in step:
                    step = step.split("DESCRIPTION")[0]
                step = step.replace("<IN LIB> ", "")

                if self.eval_set_type == "tools":
                    if data_item["dataset"] != "tools-GPT4Tools":
                        step = step.split(" ")[0]
                ret_dict["final_answer"].append(step)
        

        return ret_dict
        

