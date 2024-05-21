import os
import json


def find_final_score_files(folder, eval_type_list=[]):
    print(folder)
    print()
    exist_types = []
    if eval_type_list != []:
        for root, dirs, files in os.walk(folder):
            for type in eval_type_list:
                for file in files:
                    if file == "final_score.json":
                        if type in root:
                            exist_types.append(type)
                            json_data = json.load(open(os.path.join(root, file), "r"))
                            print(str(round((json_data["win_count"] / json_data["case_count"]) * 100, 2)), str(round((json_data["valid_case_count"] / json_data["case_count"]) * 100, 2)), str(round((json_data["win_count"] / json_data["valid_case_count"]) * 100, 2)), str(json_data["win_count"]), str(json_data["case_count"]), str(json_data["total_hallu_count"]), str(json_data["not_none_case_count"]), str(json_data["valid_case_count"]))
                            
            
    else:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file == "final_score.json":

                    json_data = json.load(open(os.path.join(root, file), "r"))
                
                    print(json_data["category_name"], str(round((json_data["win_count"] / json_data["case_count"]) * 100, 2)), str(round((json_data["valid_case_count"] / json_data["case_count"]) * 100, 2)), str(round((json_data["win_count"] / json_data["valid_case_count"]) * 100, 2)), str(json_data["win_count"]), str(json_data["case_count"]), str(json_data["total_hallu_count"]), str(json_data["not_none_case_count"]), str(json_data["valid_case_count"]))
                    


if __name__ == "__main__":
    eval_result_dir_list = ["./eval_results/wikihow_eval/chatgpt_final_eval_plan_retrieve",
                            "./eval_results/wikihow_eval/chatgpt_final_eval_task_retrieve",
                            "./eval_results/wikihow_eval/chatgpt_final_eval_select",
                            "./eval_results/wikihow_eval/chatgpt_final_eval_dfs",
                            "./eval_results/wikihow_eval/chatgpt_final_eval_rewrite",
                            "./eval_results/tools_eval/chatgpt_final_eval_plan_retrieve",
                            "./eval_results/tools_eval/chatgpt_final_eval_task_retrieve",
                            "./eval_results/tools_eval/chatgpt_final_eval_select",
                            "./eval_results/tools_eval/chatgpt_final_eval_dfs",
                            "./eval_results/tools_eval/chatgpt_final_eval_rewrite",
                            "./eval_results/robot_eval/chatgpt_final_eval_plan_retrieve",
                            "./eval_results/robot_eval/chatgpt_final_eval_task_retrieve",
                            "./eval_results/robot_eval/chatgpt_final_eval_select",
                            "./eval_results/robot_eval/chatgpt_final_eval_dfs",
                            "./eval_results/robot_eval/chatgpt_final_eval_rewrite",
                            ]
    
    eval_type_list = ["Relationships", "Family Life", "Home and Garden", "Pets and Animals", "Food and Entertaining", "Holidays and Traditions",
                      "Health", "Personal Care and Style", "Education and Communications", "Philosophy and Religion", "Sports and Fitness", "Travel",
                      "Finance and Business", "Work World", "Computers and Electronics", "Arts and Entertainment", "Hobbies and Crafts", "Youth", "Cars and Other Vehicles"]


    for folder in eval_result_dir_list:
        if "wikihow" in folder:
            find_final_score_files(folder, eval_type_list)
        else:
            find_final_score_files(folder)