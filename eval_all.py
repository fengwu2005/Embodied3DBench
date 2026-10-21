#!/usr/bin/env python3
"""
Call eval.sh for many models/tasks:
- For each MODEL, run tasks sequentially inside one thread
- Models are run in parallel (one thread per model)
No CLI args; lists are defined below.
"""
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from tqdm import tqdm
from metrics.summarize import summarize
import pandas as pd
import time

# make WORKSPACE the directory containing this script so SCRIPT path is stable
WORKSPACE = Path(__file__).resolve().parent
SCRIPT = WORKSPACE / "eval.sh"
BATCH = 100

# Define all models and tasks here
MODELS = [
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
    # "gpt5",
    # "gpt4o",
    # "claude-sonnet-4",
    # "doubao-seed-1-6-thinking",
    # "doubao-seed-1.6",
    # "doubao-1-5-thinking-vision-pro",
    # "qwen2-5-vl-72b",
    #"qwen2-5-vl-7b",
    # "internvl3-5-8b",
    # "internvl3-38b"
    #"tanyuan-qwen2-5-vl-7b-local"
    #'qwen2-5-vl-7b-local',
    'qwen2-5-vl-7b-local-pro'
    # "qwen2-5-vl-32b",
    # "qwen2-5-vl-7b-local-bench"
]

TASKS = [
         '2d_oriented_bbox_qa_detect_category',
         '2d_oriented_bbox_qa_single_part_with_desc',
         '2d_oriented_bbox_qa_single_with_desc',
         '3d_bbox_qa_single_part_with_desc',
         'obj_bbox_qa_detect_category_3d',
         'obj_bbox_qa_single_with_desc_3d',
         'correspondence_pairs_customized_2d',
         'correspondence_pairs_customized_3d',
         'correspondence_pairs_single_object_2d',
         'correspondence_pairs_single_object_3d',
         'grasp_point_2d',
         'grasp_point_3d',
         'object_point_qa_detect_category_2d',
         'object_point_qa_detect_category_3d',
         'object_point_qa_single_with_desc_2d',
         'object_point_qa_single_with_desc_3d',
         'single_func_point_2d',
         'single_func_point_3d',
         'single_func_vector_2d',
         'single_func_vector_3d',
         'single_part_point_2d',
         'single_part_point_3d',
         'spatial_rel_qa_obj2cam_distance',
         'spatial_rel_qa_obj2obj_choice',
         'spatial_rel_qa_obj2obj_distance',
         'traj_qa_prediction_2d',
         'traj_qa_prediction_3d',
    
         # 'multi_func_vector_3d',
         # 'multi_func_vector_2d',
         # 'multi_func_point_3d',
         # 'multi_func_point_2d'
]

# Number of parallel workers: one thread per model to guarantee "each model has its own thread"
MAX_WORKERS = len(MODELS)

LOG_BASE = WORKSPACE / "eval_result_26_1_26_10000_on_train_26_1_22"/ "mingxu_eval_logs"
# LOG_BASE = WORKSPACE / "eval_result_qwen25vlpro_1225_12000"/ "mingxu_eval_logs"
LOG_BASE.mkdir(parents=True, exist_ok=True)

grounding_list = ["2d_oriented_bbox_qa_detect_category",
                  "2d_oriented_bbox_qa_single_with_desc",
                  "obj_bbox_qa_detect_category_3d",
                  "obj_bbox_qa_single_with_desc_3d",
                  "2d_oriented_bbox_qa_single_part_with_desc",
                  "3d_bbox_qa_single_part_with_desc",
                  "object_point_qa_detect_category_2d",
                  "object_point_qa_detect_category_3d",
                  "object_point_qa_single_with_desc_2d",
                  "object_point_qa_single_with_desc_3d",
                  "single_part_point_2d",
                  "single_part_point_3d"]

def run_eval_for_model(model: str, pbar: tqdm, lock: Lock) -> None:
    """
    Runs all TASKS sequentially for a single model.
    Updates the provided per-model progress bar after each task.
    """
    try:
        for i in range(0,10):
            for task in TASKS:
                if task in grounding_list and i >= 5:
                    continue
                out_dir = LOG_BASE / task
                out_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = out_dir / f"{model}_{task}_{i}.out.log"
                stderr_path = out_dir / f"{model}_{task}_{i}.err.log"
    
                cmd = [
                    "bash",
                    str(SCRIPT),
                    "--model-name",
                    model,
                    "--task",
                    task,
                    "--out",
                    f"./eval_result_26_1_26_10000_on_train_26_1_22/{model}/{task}/{i}",
                    # f"./eval_result_qwen25vlpro_1225_12000/{model}/{task}/{i}",
                    "--n",
                    f"{BATCH*i}",
                ]
                print("cmd:", cmd)
                # run and capture logs to files; set cwd to WORKSPACE
                try:
                    with open(stdout_path, "w", encoding="utf-8") as stdout_f, \
                         open(stderr_path, "w", encoding="utf-8") as stderr_f:
                        subprocess.run(cmd, check=True, stdout=stdout_f, stderr=stderr_f, cwd=str(WORKSPACE))
                except subprocess.CalledProcessError:
                    # update bar to reflect progress before re-raising so display stays consistent
                    # DO NOT acquire the same lock here; tqdm already uses the lock set by tqdm.set_lock(lock)
                    pbar.update(1)
                    pbar.refresh()
                    raise
    
                # update progress bar for this model (do not hold external lock — tqdm handles locking)
            pbar.update(1)
            pbar.refresh()
            # print(f"{model} ok {i}")
            
    except Exception:
        # ensure bar is left in consistent state on error
        try:
            pbar.close()
        except Exception:
            pass
        raise
		
def main():
    if not SCRIPT.exists():
        print(f"eval.sh not found at {SCRIPT}", file=sys.stderr)
        sys.exit(2)

    # prepare lock for tqdm thread-safety
    lock = Lock()
    tqdm.set_lock(lock)

    # prepare one progress bar per model (position controls vertical ordering)
    bars = {}
    for i, model in enumerate(MODELS):
        bars[model] = tqdm(total=10, desc=model, position=i, leave=True, mininterval=0.1)

    # submit one thread per model; each thread runs all tasks sequentially
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(run_eval_for_model, model, bars[model], lock): model for model in MODELS}
        try:
            failures = []
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    fut.result()
                    print(f"[OK]  model={model}")
                except subprocess.CalledProcessError as e:
                    print(f"[FAIL] model={model} exit={e.returncode}", file=sys.stderr)
                    failures.append((model, e.returncode))
                    # do not shutdown immediately; continue other models
                except Exception as e:
                    print(f"[ERR]  model={model} {e}", file=sys.stderr)
                    failures.append((model, 1))
                    # continue other models
            # after all finished, exit non-zero if any failures
            if failures:
                print(f"[SUMMARY] failures={len(failures)}", file=sys.stderr)
                sys.exit(1)
        finally:
            # close all bars
            for b in bars.values():
                try:
                    b.close()
                except Exception:
                    pass

    # summarize results in a data frame
    # summary_df = pd.DataFrame()
    # for model in MODELS:
    #     for task in TASKS:
    #         result_dir = WORKSPACE / "fengwu_result" / model / task
    #         # result_dir have a json file, summarize the json file
    #         json_files = sorted(list(result_dir.glob("*.json")))
    #         if json_files:
    #             # print(json_files[0])
    #             mean_metrics = summarize(json_files[-1])
    #             # add mean_metrics to a data frame
    #             # print(mean_metrics)
    #             summary_df.loc[model, task] = mean_metrics
    # print(summary_df)
    # # save the data frame to csv
    # summary_df.to_csv(WORKSPACE / "fengwu_eval_summary.csv")

if __name__ == "__main__":
    main()