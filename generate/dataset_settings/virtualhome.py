"""
@author: Guo Shiguang
@software: PyCharm
@file: virtualhome.py
@time: 2024/1/30 21:01
@description: 
"""
import numpy as np

from generate.dataset_settings.base_dataset import DataItem, TaskDataset


class VirtualHomeDataset(TaskDataset):
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func, predict_path):
        super().__init__(task_data, action_lib, action_emb, model_func, predict_path)

    def get_no_constraints_actions(self, task: DataItem, temp=1.0) -> list[str] or None:
        title = task.title
        method = task.method
        prompt = [
            "<Task>: Write an email\n<Method>: You are a robot in your home. Your movement needs to be completed by a series of very simple actions.",
            "1. Walk to home office\n2. Walk to computer\n3. Find computer\n4. Turn to computer\n5. Look at computer\n6. Walk to computer\n7. Find chair\n8. Sit on chair\n9. Find keyboard\n10. Grab keyboard\n11. Find mouse\n12. Grab mouse\n13. Type on keyboard",
            "<Task>: Take shower\n<Method>: You are a robot in your home. Your movement needs to be completed by a series of very simple actions.",
            "1. Find clothes dress\n2. Find towel\n3. Walk to bathroom\n4. Walk to shower\n5. Find shower",
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
