"""
@author: Guo Shiguang
@software: PyCharm
@file: base_dataset.py
@time: 2024/1/30 20:59
@description: 
"""
import faiss
import numpy as np


class DataItem:
    # 为了可以自动补全
    def __init__(self, title: str,
                 steps: list[str],
                 method: str | None = None,
                 file_name: str = None,
                 cate_hierarchy: list[str] = None
                 ):
        self.file_name: str = file_name
        self.title: str = title
        self.method: str = method
        self.steps: list[str] = steps
        self.cate_hierarchy: list[str] = cate_hierarchy


class TaskDataset:
    def __init__(self, task_data: list[dict], action_lib: list[str], action_emb: np.ndarray, model_func, predict_path):
        self.task_data: list[DataItem] = [DataItem(**item) for item in task_data]
        self.action_lib: list[str] = action_lib
        self.action_emb: faiss.IndexFlatIP = faiss.IndexFlatIP(action_emb.shape[1])
        self.action_emb.add(action_emb)
        self.model_func = model_func
        self.predict_path = predict_path
        self.init_plan_instruction = "You will be given a task and a method to complete the task. If no method is specified it will be set to \"None\". You need to generate a plan that satisfies the given tasks and methods. The plan needs to be a list of several actions and each action should be a complete and short sentence separated by newlines. Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3..."
        self.task_retrieve_instruction = (
            "You will be given a task, a method to complete the task and several available actions. If no method is specified it will be set to \"None\". Here are a few things you need to keep in mind:\n"
            "1. You need to generate a plan that satisfies the given tasks and methods.\n"
            "2. You can only use the steps in <Actions in Library> to complete a given task, even though the provided steps may not complete the task.\n"
            "3. You must use the actions in the library exactly. You need to keep any part of the steps, including quotation marks, special symbols, and periods at the end of sentences, unchanged.\n"
            "Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3...")
        self.plan_retrieve_instruction = (
            "You will be given a task, a method to complete the task, an initial plan to follow and a number of available actions. If no method is specified it will be set to \"None\".Here are a few things you need to keep in mind:\n"
            "1. You need to generate a plan that satisfies the given tasks and methods.\n"
            "2. The initial plan is a reference only, which means you should not output the steps in the initial plan directly.\n"
            "3. You can only use the steps in Actions in Library to complete a given task, even though the provided steps may not complete the task.\n"
            "4. You must use the actions in the library exactly. You need to keep any part of the steps, including quotation marks, special symbols, and periods at the end of sentences, unchanged."
            "Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3...")
        self.select_initial_step_instruction = (
            "You will be given a task and a method to complete the task. If no method is specified it will be set to \"None\". Remember:\n"
            "1. Based on your plan for a given task and method, you need to generate the first step of completing the task with the specified method. The step you generated will be the initial step in planning. You need to pay attention to scalability in this step.\n"
            "2. There is no going back in your generating process, you cannot try to delete or modify a previously existing step.\n"
            "Send your answer in the following format: [New step: step]")
        self.select_next_step_instruction = (
            "You will be given a task, a method to complete the task and a current plan. If no method is specified it will be set to \"None\". If the current plan is empty, the plan will also be set to \"None\". Remember:\n"
            "1. You need to generate the next step in the plan that needs to meet the given tasks and methods based on the existing plan. Please note that the step you generate will be added to the end of the existing steps, and you need to pay attention to maintain the coherence of the overall steps.\n"
            "2. There is no going back in your generating process, you cannot try to delete or modify a previously existing step.\n"
            "3. If you think the current plan is sufficient for the task, just output \"[New step: None]\". You only need to output one step and do nothing else.\n"
            "Send your answer in the following format: [New step: step] or [New step: None]")
        self.select_select_instruction = (
            "You will be given a task, a method to complete the task, a current plan and several candidate actions. Candidate actions are called <Actions in Library>. If no method is specified it will be set to \"None\". If the current plan is empty, the plan will also be set to \"None\". Here are a few things you need to keep in mind:\n"
            "1. You need to select the next step in the plan from the candidate actions that satisfies the given task and method based on the currently existing plan. Please note that the step you select will be added to the end of the existing steps, and you need to pay attention to maintain the coherence of the overall steps.\n"
            "2. There is no going back in your generating process.\n"
            "3. You can only use the steps in <Actions in Library> to complete a given task. If you think that the provided steps may not accomplish the task, you need to select \"[New step: None of these]\". Never use your own steps to complete a task, all output needs to be selected from the options provided.\n"
            "4. You must use the actions in the library exactly. You can't just output the sequence number of an action, you must output the entire sentence. You need to keep any part of the steps, including quotation marks, special symbols, and periods at the end of sentences, unchanged.\n"
            "Send your answer in the following format and do nothing else: [New step: None of these] or [New step: step]")  # none here means reject
        self.dfs_initial_step_instruction = (
            "You will be given a task and a method to complete the task. If no method is specified it will be set to \"None\". Remember:\n"
            "1. Based on your plan for a given task and method, you need to generate three candidates for the first step of completing the task with the specified method. These three steps must be different. I would choose one of the three candidates you generated as an initial step in planning. You need to pay attention to scalability in this step.\n"
            "2. There is no going back in your generating process, you cannot try to delete or modify a previously existing step.\n"
            "Send your answer in the following format: [1. candidate1\n2. candidate2\n3. candidate3]")
        self.dfs_next_step_instruction = (
            "You will be given a task, a method to complete the task and a current plan. If no method is specified it will be set to \"None\". If the current plan is empty, the plan will also be set to \"None\". Remember:\n"
            "1. You need to generate three candidates for the next step of the current plan based on the existing plan to satisfy the plan for the given task and method. These three steps must be different. Note that I will add one of these three candidates you generated to the end of the existing step. You need to pay attention to maintaining the coherence of the overall steps.\n"
            "2. There is no going back in your generating process, you cannot try to delete or modify a previously existing step.\n"
            "3. If you think the current plan is sufficient for the task, just output \"[None]\". You only need to output steps and do nothing else.\n"
            "Send your answer in the following format: [1. candidate1\n2. candidate2\n3. candidate3] or [None]")
        self.dfs_select_instruction = (
            "You will be given a task, a method to complete the task, a current plan and several candidate actions. Candidate actions are called <Actions in Library>. If no method is specified it will be set to \"None\". If the current plan is empty, the plan will also be set to \"None\". Here are a few things you need to keep in mind:\n"
            "1. You need to select the next step in the plan from the candidate actions that satisfies the given task and method based on the currently existing plan. Please note that the step you select will be added to the end of the existing steps, and you need to pay attention to maintain the coherence of the overall steps.\n"
            "2. There is no going back in your generating process.\n"
            "3. You can only use the steps in <Actions in Library> to complete a given task. If you think that the provided steps may not accomplish the task, you need to select \"[New step: None of these]\". Never use your own steps to complete a task, all output needs to be selected from the options provided.\n"
            "4. You must use the actions in the library exactly. You can't just output the sequence number of an action, you must output the entire sentence. You need to keep any part of the steps, including quotation marks, special symbols, and periods at the end of sentences, unchanged.\n"
            "Send your answer in the following format and do nothing else: [New step: None of these] or [New step: step]")  # none here means rollback
        self.refine_initial_plan = (
            "You will be given a task, a method to complete the task and several actions for reference. These actions are called <References>. If no method is specified, it will be set to \"None\".Remember:\n"
            "1. You need to refer to the content in <References> to generate a plan that can complete the task in the specified method. \n"
            "2. The generated plan does not need to use the exact steps in <Reference>. You can generate any plan as long as it can complete the task in the specified method. In subsequent operations, I will use other actions in the library to rewrite it, so the plan you generate needs to be as consistent in style as possible with these actions.\n"
            "Send your answer in the following format and do nothing else: 1. step1\n2. step2\n3. step3...")
        self.refine_instruction = (
        "You will be given a task, a method to complete the task, a current plan and several candidate actions. Candidate actions are called <Actions in Library>. If no method is specified it will be set to \"None\". If the current plan is empty, the plan will also be set to \"None\".\n"
        "Use the actions listed below to refine your current steps to complete your task. Actions marked with <TO BE REPLACED> indicate that the content was not found in the action library, and actions marked with <IN LIB> indicate that they are in the action library. You need to analyze which actions in the provided action library can be added to the action list and replace some or all of the actions marked with <TO BE REPLACED>. We encourage you to add more <TO BE REPLACED> content to complete these steps.\n"
        "You can do the following:\n"
        "* Replace any number of <TO BE REPLACED>-like operations with any number of <IN LIB> operations.\n"
        "* Replace any number of <IN LIB> operations with any number of <IN LIB> operations as the latter are better suited to the task.\n"
        "* Replace any number of <IN LIB> operations with more general <IN LIB> operations.\n"
        "* Insert any number of <IN LIB> operations that differ from existing steps.\n"
        "* Insert any number of <TO BE REPLACED> operations to fill in missing content between steps.\n"
        "* Remove any number of redundant <IN LIB> operations.\n"
        "* Remove any number of redundant <TO BE REPLACED> operations.\n"
        "* Remove any number of overly verbose <IN LIB> operations.\n"
        "* Compare several similar <IN LIB> operations and select the best one to add to the operations list.\n"
        "* Other reasonable actions.\n"
        "Remember:\n"
        "1. Your output needs to be a list, and each element in the list needs to start with \"{<IN LIB>\" or \"{<TO BE REPLACED>\" and end with \"}\".\n"
        "2. The rewritten sentences need to cover roughly the same scope as before the rewrite, although they may differ in detail.\n"
        "3. Only actions that are newly added from <Actions in library> can be marked with <IN LIB>, otherwise they need to maintain their attributes in <Current steps>.\n"
        "4. You must use the actions in the library exactly. You can't just output the sequence number of an action, you must output the entire sentence. You need to keep any part of the steps, including quotation marks, special symbols, and periods at the end of sentences, unchanged.\n"
        "5. Always output the complete plan, not only newly added or changed steps. If the content containing <IN LIB> is still needed, you need to output it completely.\n"
        "Send your answer in the following format and do nothing else: [1. {<IN LIB> step1}\n2. {<TO BE REPLACED> step2}\n...]")

        self.refined_2_shot_examples = [
            "<Task>: How to Increase Your Income\n<Method>: Cutting Down on Expenses\n<Current steps>:\n1. {<TO BE REPLACED> Avoid eating out.}\n2. {<TO BE REPLACED> Cancel unused subscriptions and memberships.}\n3. {<TO BE REPLACED> Bike or walk to work, rather than drive.}\n4. {<TO BE REPLACED> Find free or low-cost entertainment options instead of expensive outings.}\n5. {<TO BE REPLACED> Reduce your rent.}\n\n<Actions in library>:\n{<IN LIB> Cancel or suspend memberships or subscriptions that you're no longer using, or that you're using ineffectively.}\n{<IN LIB> Select Cancel Subscription.}\n{<IN LIB> Cancel your dating profiles and subscriptions.}\n{<IN LIB> Click Cancel Subscription.}\n{<IN LIB> Click Cancel subscription.}\n{<IN LIB> Find fun things to do together as a family that don't cost a lot.}\n{<IN LIB> Find alternative or more cost effective ways to spending time on your own.}\n{<IN LIB> Pursue less costly hobbies.}\n{<IN LIB> Choose affordable activities.}\n{<IN LIB> Take advantage of free fun.}",
            "1. {<TO BE REPLACED>Avoid eating out.}\n2. {<IN LIB> Cancel or suspend memberships or subscriptions that you're no longer using, or that you're using ineffectively.}\n3. {<TO BE REPLACED> Bike or walk to work, rather than drive.}\n4. {<IN LIB> Choose affordable activities.}\n5. {<TO BE REPLACED> Reduce your rent.}",
            '<TASK>: How to Make Fried Chicken with Buttermilk and Tarragon\n<Method>: None\n<Current steps>:\n1. {<TO BE REPLACED> Marinate chicken pieces in buttermilk and tarragon for at least 4 hours or overnight in the refrigerator.}\n2. {<TO BE REPLACED> Remove chicken from the buttermilk marinade and let excess drip off.}\n3. {<TO BE REPLACED> Dredge the chicken in seasoned flour, ensuring it is evenly coated.}\n4. {<TO BE REPLACED> Heat oil in a pan to 350°F (175°C) and carefully place the chicken in the hot oil.}\n5. {<TO BE REPLACED> Fry until golden brown and fully cooked, usually about 15-20 minutes depending on the size of the chicken pieces.}\n6. {<TO BE REPLACED> Once cooked, place the fried chicken on a wire rack to drain excess oil.}\n7. {<TO BE REPLACED> Serve and enjoy!}\n\n<Actions in library>:\n{<IN LIB> Refrigerate marinated chicken for 4 to 6 hours.}\n{<IN LIB> Allow the chicken to marinate overnight.}\n{<IN LIB> Shake off the marinade from the chicken pieces.}\n{<IN LIB> Remove chicken pieces from the buttermilk mixture and dredge in flour.}\n{<IN LIB> Add chicken pieces to seasoned flour and toss to coat.}\n{<IN LIB> Heat oil in skillet to 350 °F (177 °C).}\n{<IN LIB> Preheat the oil to 350° Fahrenheit or 176° Celsius.}\n{<IN LIB> Fry the chicken in a frying pan until crispy and golden in color.}\n{<IN LIB> Fry about 2 to 3 minutes or until they are golden brown and crispy.}',
            '1. {<TO BE REPLACED> Mix buttermilk, tarragon, salt and pepper in a large bowl.}\n2. {<TO BE REPLACED> Add chicken pieces to mixture to marinate.}\n3. {<IN LIB> Refrigerate marinated chicken for 4 to 6 hours.}\n4. {<IN LIB> Remove chicken pieces from the buttermilk mixture and dredge in flour.}\n5. {<IN LIB> Heat oil in skillet to 350 °F (177 °C).}\n6. {<IN LIB> Fry the chicken in a frying pan until crispy and golden in color.}\n7. {<TO BE REPLACED> Once cooked, place the fried chicken on a wire rack to drain excess oil.}\n8. {<TO BE REPLACED> Serve and enjoy!}']
    def feedback(self, task: DataItem, past_retrieved_steps: list[str], current_plan: list[str]):
        raise NotImplementedError

    @staticmethod
    def check_init_plan(no_constraints_actions_ans: str) -> list[str] or None:
        def format_no_con_actions(no_constraints_actions_ans):
            check_redundance = no_constraints_actions_ans.split("\n\n")
            if len(check_redundance) > 1:
                no_constraints_actions_ans = max(check_redundance, key=len)
            return no_constraints_actions_ans.split("\n")

        no_constraints_actions_list = format_no_con_actions(no_constraints_actions_ans)
        no_constraints_actions = [item.replace(item.split(".")[0] + ".", "").strip() for item in
                                  no_constraints_actions_list]
        if len(no_constraints_actions) == 0:
            return None
        return no_constraints_actions

    def extract_action(self, action: str):
        max_loop=3
        loop=0
        while action.startswith("{") or action.split(".")[0].isdigit():
            loop+=1
            if loop>max_loop:
                break
            if action.startswith("{"):
                action = action[1:-1]
            action = (action.replace("<TO BE REPLACED> ", "")
                      .replace("<IN LIB> ", "")
                      .replace("<TO BE REPLACED>", "")
                      .replace("<IN LIB>", "")
                      )
            if action.split(".")[0].isdigit():
                action = action.replace(action.split(".")[0] + ".", "")
            action = action.strip()
        return action.strip()

    def get_no_constraints_actions(self, task: DataItem, temp=1.0) -> list[str] or None:
        raise NotImplementedError

    @staticmethod
    def check_list_ans(output: str, ) -> list[str] or None:
        if "\n\n" in output:
            output_list = output.split("\n\n")
            cand_list = []
            for cand in output_list:
                if all([item.split(".")[0].isdigit() for item in cand.split("\n") if item.strip() != ""]):
                    cand_list.append(cand)
            if len(cand_list) != 1:
                return None
            output = cand_list[0]
        if all([item.split(".")[0].isdigit() for item in output.split("\n") if item.strip() != ""]):
            return [item.strip() for item in output.split("\n") if item.strip() != ""]
        else:
            return None

    @staticmethod
    def check_select_ans(output: str) -> str or None:
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        if "\n\n" in output:
            output_list = output.split("\n\n")
            cand_list = []
            for cand in output_list:
                if cand.startswith("New step:"):
                    cand_list.append(cand)
            if len(cand_list) != 1:
                return None
            output = cand_list[0]
        if output.startswith("New step:"):
            output = output.replace("New step:", "").strip()
            if output == "":
                return None
            return output
        else:
            return None

    @staticmethod
    def check_dfs_candidates_ans(output: str) -> list[str] or None:
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        if "\n\n" in output:
            output_list = output.split("\n\n")
            cand_list = []
            for cand in output_list:
                if all([item.split(".")[0].isdigit() for item in cand.split("\n") if item.strip() != ""]):
                    cand_list.append(cand)
            if len(cand_list) != 1:
                return None
            output = cand_list[0]
        if all([item.split(".")[0].isdigit() for item in output.split("\n") if item.strip() != ""]):
            return [item.strip() for item in output.split("\n") if item.strip() != ""]
        else:
            return None

    @staticmethod
    def check_refine_ans(output) -> list[str] or None:
        def inner_check(item):
            if item.split(".")[0].isdigit():
                item= item.replace(item.split(".")[0] + ".", "").strip()
            return (item.startswith("{<IN LIB>") or item.startswith("{<TO BE REPLACED>")) and item.endswith("}")

        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        if "\n\n" in output:
            output_list = output.split("\n\n")
            cand_list = []
            for cand in output_list:
                if all([inner_check(item) for item in cand.split("\n") if item.strip() != ""]):
                    cand_list.append(cand)
            if len(cand_list) != 1:
                return None
            output = cand_list[0]
        if not output.startswith("1."):
            return None
        if all([inner_check(item) for item in output.split("\n") if item.strip() != ""]):
            return [item.strip() for item in output.split("\n") if item.strip() != ""]
        else:
            return None
