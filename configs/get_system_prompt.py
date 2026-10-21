import os
from pathlib import Path
import json as _json

task_response_formats_path = Path(__file__).parent.joinpath("task_response_formats.json")
if task_response_formats_path.exists():
    with open(task_response_formats_path, "r", encoding="utf-8") as f:
        task_response_formats = _json.load(f)
system_base = (
    "You are an evaluation model for spatial reasoning QA tasks.\n"
    "Produce deterministic, machine-parseable outputs for spatial/function reasoning questions.\n\n"
    "RESPONSE CONTRACT:\n"
    "- Your final output must consist of exactly one triple-backtick fenced JSON block labeled `json` without any additional texts or parts.\n"
    # "- Your final output must consist of exactly one optional <think></think> block followed by one triple-backtick fenced JSON block labeled `json` without any additional texts or parts.\n"
    # "- Put all non-JSON text inside <think></think> to keep the format clean.\n"
    "- Copy the EXAMPLE VALUE into the JSON block to keep the format, then edit values to be your answer (ensure at least one meaningful answer).\n"
    # "- The <think> block must include the inferred `task` and `function_type` and a one-line rationale.\n"
    "- When you answer 3D questions or distance, the output unit is m.\n\n"
    "- Don't output any thinking process, just the JSON answer."
    # "When calling this adapter include the parameter `task_name` so the adapter can append the exact expected JSON `value` example for that task.\n"
)

def get_system_prompt(task_name: str) -> (bool, str):
    task_format = task_response_formats.get(task_name, None)
    if not task_format:
        raise ValueError(f"Task name '{task_name}' not found in task_response_formats.json")

    task_desc = task_format.get("description", "")
    task_example = task_format.get("value_example")
    try:
        example_text = _json.dumps(task_example, ensure_ascii=False, indent=2)
    except Exception:
        example_text = str(task_example)

    example_text_fenced = "```json\n" + example_text + "\n```"
    return (
        system_base
        + f"\nREQUIRED VALUE FORMAT FOR TASK {task_name}:\n{task_desc}\nEXAMPLE VALUE:\n"
        + example_text_fenced
    )