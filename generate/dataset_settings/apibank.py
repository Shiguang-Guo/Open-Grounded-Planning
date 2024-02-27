"""
@author: Guo Shiguang
@software: PyCharm
@file: apibank.py
@time: 2024/1/30 21:01
@description: 
"""
import numpy as np

from generate.dataset_settings.base_dataset import DataItem, TaskDataset
from utils.chat import get_ans

class APIBankDataset(TaskDataset):
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func,predict_path):
        super().__init__(task_data, action_lib, action_emb, model_func,predict_path)
        self.init_plan_instruction = "You will be given a task and a method to complete the task. If no method is specified it will be set to \"None\". Your task is to generate a plan that satisfies the given tasks and methods. The plan needs to be a list of several apis and each api should have an api name and following by a function-comment-style description separated by newlines. Each line of description needs to start with the function name, followed by \"DESCRIPTION\", and finally the content of the comment. Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3..."

    def get_no_constraints_actions(self, task:DataItem, temp=1.0) -> list[str] or None:
        title=task.title
        method=task.method
        prompt = [
            "<Task>: Check to see if I have set a reminder to email a friend on March 16th at 10am\n<Method>: One or two steps are usually enough to complete the task, and there are only a few cases where more may be required.",
            "1. GetUserToken DESCRIPTION: Get user token\n2. QueryReminder DESCRIPTION: Get a list of all reminders",
            "<Task>: Can you turn on my coffee maker at 7 am tomorrow?\n<Method>: One or two steps are usually enough to complete the task, and there are only a few cases where more may be required.",
            "1. TimedSwitch DESCRIPTION: scheduling a timed switch. Sets a switch-time for the specified device-id and executes an on/off command at the specified time.",
            f"<Task>: {title}\n<Method>: {method}"]

        while no_constraints_actions_ans := self.model_func(query=prompt, temperature=temp, instruct=self.init_plan_instruction,save_path=f"{self.predict_path}_chat/{task.file_name}"):
            no_constraints_actions = self.check_init_plan(no_constraints_actions_ans)
            if no_constraints_actions is not None:
                return no_constraints_actions
        return None
