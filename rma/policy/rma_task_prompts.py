"""RoboMemArena (RMA) task prompts, keyed ``task<N>`` for N in 1..26.

Lifted verbatim from the RMA evaluation harness
(``scratch/rma_structure/evalcode/openpi_minimal_runtime_task_prompts.py``)
so the trainer-format conversion uses byte-identical prompts to the eval-time
prompts.  Do NOT reword these: train/eval prompt strings must match exactly.
"""

from __future__ import annotations

TASK_PROMPTS: dict[str, str] = {
    "task1": "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
    "task2": "Pick and place butter into the basket, then pick and place popcorn into the same basket.",
    "task3": "Pick and place cream into the basket, then pick and place chocolate into the same basket.",
    "task4": "Open and close all drawers in order to check. Put butter into the drawer that already contains an object.",
    "task5": "Open and close all drawers in order to check. Put butter into the empty drawer.",
    "task6": "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.",
    "task7": "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.",
    "task8": "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    "task9": "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    "task10": "Pour wine into the mug twice.",
    "task11": "Put cookies into the top drawer and put butter into another drawer.",
    "task12": "Put cookies into the middle drawer and then put chocolate into the same drawer.",
    "task13": "Put cookies into the middle drawer and then put butter into the same drawer.",
    "task14": "Put cookies into the top drawer and put chocolate into another drawer.",
    "task15": "Pick and place butter into the frypan, then pour milk over it twice.",
    "task16": "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.",
    "task17": "Put butter into the middle drawer and then put chocolate into the same drawer.",
    "task18": "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.",
    "task19": "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2.",
    "task20": "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
    "task21": "Put butter into the microwave and then put chocolate into the location where the butter was placed.",
    "task22": "Pour tomato sauce over cookies twice, then put the cookies into the microwave.",
    "task23": "Put cream into the microwave and then put popcorn into the location where the cream was placed.",
    "task24": "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.",
    "task25": "Pick and place butter and cream from plate1 to plate2, respectively.",
    "task26": "Pick and place chocolate and cream from plate1 to plate2, respectively.",
}


def get_prompt(task_id: str, fallback_task_name: str = "") -> str:
    """Prompt for ``task<N>``; falls back to the task dir slug with '_' -> ' '."""
    if task_id in TASK_PROMPTS:
        return TASK_PROMPTS[task_id]
    return fallback_task_name.replace("_", " ")
