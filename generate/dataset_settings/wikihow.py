"""
@author: Guo Shiguang
@software: PyCharm
@file: wikihow.py
@time: 2024/1/30 21:00
@description: 
"""
import numpy as np

from generate.dataset_settings.base_dataset import DataItem, TaskDataset


class WikihowDataset(TaskDataset):
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func, predict_path):
        super().__init__(task_data, action_lib, action_emb, model_func, predict_path)


    def get_no_constraints_actions(self, task:DataItem, temp=1.0) -> list[str] or None:
        title=task.title
        method=task.method
        prompt = [
            "<Task>: How to Watch Disney Plus on iPhone\n<Method>: None",
            "1. Open the Disney+ app.\n2. Tap START FREE TRIAL to sign up.\n3. Enter your email address.\n4. Scroll down and tap AGREE & CONTINUE.\n5. Enter a password and tap SIGN UP.\n6. Tap a billing option.\n7. Follow the on-screen instructions to confirm.\n8. Check out what's available.\n9. Tap a movie or show.\n10. Tap Play to start watching.",
            "<Task>: How to Improve Your Posture\n<Method>: Using Exercise to Improve Your Posture",
            "1. Improve your core muscles with deep abdominal stretching.\n2. Do a shoulder blade squeeze.\n3. Train your muscles for better posture with strength training.\n4. Pretend you're a penguin to stretch your shoulders.\n5. Use stretching for a sore neck or back.\n6. Practice yoga to increase flexibility and help with posture.",
            f"<Task>: {title}\n<Method>: {method}"
            ]

        max_rewrite_loop_times = 10
        current_rewrite_loop_times = 0
        while no_constraints_actions_ans := self.model_func(query=prompt, temperature=temp,
                                                            instruct=self.init_plan_instruction,
                                                            save_path=f"{self.predict_path}_chat/{task.file_name}"):
            no_constraints_actions = self.check_init_plan(no_constraints_actions_ans)
            if no_constraints_actions is not None and no_constraints_actions != [""]:
                return no_constraints_actions
            current_rewrite_loop_times += 1
            if current_rewrite_loop_times >= max_rewrite_loop_times:
                raise ValueError("chatgpt不生成")
        return None
