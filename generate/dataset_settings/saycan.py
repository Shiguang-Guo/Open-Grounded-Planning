"""
@author: Guo Shiguang
@software: PyCharm
@file: saycan.py
@time: 2024/1/30 21:01
@description: 
"""
import numpy as np

from generate.dataset_settings.base_dataset import DataItem, TaskDataset


class SaycanDataset(TaskDataset):
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func,
                 predict_path
                 ):
        super().__init__(task_data, action_lib, action_emb, model_func, predict_path)

    def get_no_constraints_actions(self, task:DataItem, temp=1.0) -> list[str] or None:
        title=task.title
        method=task.method
        prompt = [
            "<Task>: I need a cake\n<Method>: As a robot with only one gripper, you are surrounded by a far counter, a near counter, a table, and a trash can. You are located near the table. You can only perform one action at a time, such as moving or picking up and putting down. Environmental status:cake on far counter""",
            "1. go to the far counter\n2. find a cake\n3. pick up the cake\n4. bring it to you\n5. put done the cake",
            "<TASK>: visit the far counter and the near counter\n<METHOD>: As a robot with only one gripper, you are surrounded by a far counter, a near counter, a table, and a trash can. You are located near the table. You can only perform one action at a time, such as moving or picking up and putting down. Environmental status:navigate only",
            """1. go to the far counter\n2. go to the near counter""",
            f"<Task>: {title}\n<Method>: {method}"]

        max_rewrite_loop_times = 10
        current_rewrite_loop_times = 0
        while no_constraints_actions_ans := self.model_func(query=prompt, temperature=temp,
                                                            instruct=self.init_plan_instruction,
                                                            save_path=f"{self.predict_path}_chat/{task.file_name}"):
            no_constraints_actions = self.check_init_plan(no_constraints_actions_ans)
            if no_constraints_actions is not None:
                return no_constraints_actions
            current_rewrite_loop_times += 1
            if current_rewrite_loop_times >= max_rewrite_loop_times:
                raise ValueError("chatgpt不生成")
        return None
