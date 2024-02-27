"""
@author: Guo Shiguang
@software: PyCharm
@file: gpt4tools.py
@time: 2024/1/30 21:09
@description: 
"""
import numpy as np

from generate.dataset_settings.base_dataset import DataItem, TaskDataset


class GPT4ToolsDataset(TaskDataset):
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func,predict_path):
        super().__init__(task_data, action_lib, action_emb, model_func,predict_path)
        self.init_plan_instruction = "You will be given a task and a method to complete the task. If no method is specified it will be set to \"None\". Your task is to generate a plan that satisfies the given tasks and methods. The plan needs to be a list of several apis and each api should have an api name and following by a function-comment-style description separated by newlines. Each line of description needs to start with the function name, followed by \"DESCRIPTION\", and finally the content of the comment. Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3..."

    def get_no_constraints_actions(self, task:DataItem, temp=1.0) -> list[str] or None:
        title=task.title
        method=task.method
        prompt = [
            "<Task>: Generate an image of a happy dog with an umbrella\n<Method>: One or two steps are usually enough to complete the task, and there are only a few cases where more may be required.",
            "1. Generate Image From User Input Text DESCRIPTION: useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects.",
            "<Task>: Generate a real image of a boat from the depth map\n<Method>: One or two steps are usually enough to complete the task, and there are only a few cases where more may be required.",
            "1. Predict Depth On Image DESCRIPTION: useful when you want to detect depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. \n2. Generate Image Condition On Depth DESCRIPTION: useful when you want to generate a new real image from both the user description and depth image. like: generate a real image of a object or something from this depth image, or generate a new real image of a object or something from the depth map.",
            f"<Task>: {title}\n<Method>: {method}"]

        while no_constraints_actions_ans := self.model_func(query=prompt, temperature=temp, instruct=self.init_plan_instruction,save_path=f"{self.predict_path}_chat/{task.file_name}"):
            no_constraints_actions = self.check_init_plan(no_constraints_actions_ans)
            if no_constraints_actions is not None:
                return no_constraints_actions
        return None
