eval_compare_plan_prompt="""For a given task, a possible method to solve the task and two corresponding plans, please evaluate the plans based on three aspects: completeness, feasibility, and relevance to the task, with a maximum score of 10 points for each aspect. The detailed scoring criteria for each aspect are as follows:
- Completeness: Examining whether the plan is comprehensive, with a focus on the coherence between steps, the presence of necessary steps, and the avoidance of arbitrarily introduced conditions.
- Feasibility: Assessing the practicality of the plan, considering whether each step can be implemented, whether the plan aligns with common sense, adheres to human ethical standards, and avoids excessive redundant steps.
- Relevance to the task: Evaluating the extent to which the plan is related to the given task, considering the use of provided task conditions and whether it achieves the desired goals of the task.

Task: {TASK}
Method: {METHOD}

Plan1:
{PLAN1}

Plan2:
{PLAN2}

Now, read the task, method and plans provided, and compare the plans. In the 'Analysis' section, provide a brief rationale for your comparison in the three aspects. Then, based on your analyses, choose the better plan (could be chosen from [Plan1, Plan2]):

Output:

<Analysis>
- Completeness: 
- Feasibility: 
- Relevance:
</Analysis>

<Better Plan> [Plan1, Plan2] </Better Plan>

"""
