"""QA retrieval helper for evaluation.

This module provides QALoader which:
 - Initializes with a root directory containing task folders with QA files.
 - Builds a deduplicated sampled evaluation set for a given task and saves
   it as a JSON mapping: {"<task_name>": ["<qa_path>", ...]}.
 - Loads the previously saved mapping and returns the list of QA objects
   (each QA file is JSON-loaded) or the raw mapping.

Example
-------
from evaluator.constructors.qa_loader import QALoader
loader = QALoader(r"D:/data/qa_root")
mapping = loader.build_eval_set("task_0", num_samples=50, seed=42)
qa_list = loader.load_eval_qa_list(mapping_path=None, task_name="task_0")
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)
 

class QALoader:
	"""QA retrieval class.

	Args:
		root: root directory where task subfolders live. For a task named
			"task_0" its QA files are expected under ``root/task_0``.
		out: optional output directory to save mappings. If not provided,
			defaults to the same path as `root`.
	"""

	def __init__(self, root: str, out: Optional[str] = None) -> None:
		self.root = Path(root)
		self.out = Path(out) if out is not None else self.root
		if not self.root.exists():
			logger.warning("QALoader root %s does not exist yet", self.root)
		if not self.out.exists():
			os.makedirs(self.out, exist_ok=True)

	def _gather_files_for_task(self, task_name: str, desired_count: Optional[int] = None) -> List[Path]:
		"""Optimized fast sampler for files under root/task_name (headless safe).

		If `desired_count` is provided and is less than or equal to the number of
		top-level entries (secondary directories), this function will preferentially
		sample at most one file per entry until `desired_count` is reached to
		maximize diversity across entries.
		"""
		task_dir = self.root / task_name
		files: List[Path] = []

		# MAX_CANDIDATES = 200
		# MAX_DIRS_SAMPLE = 800
		MAX_CANDIDATES = 100
		MAX_DIRS_SAMPLE = 100

		# ✅ Step 1. Fast path: common known layout (outputs/<task_name>/...)
		fast_path = self.root / "outputs" / task_name
		base_dirs: List[Path] = []

		if fast_path.exists():
			base_dirs.append(fast_path)
		elif task_dir.exists():
			base_dirs.append(task_dir)
		else:
			# As a last resort, limit search depth to avoid global recursive scan
			logger.debug("Searching shallow dirs under %s for task_name=%s", self.root, task_name)
			# for dirpath, dirnames, _ in os.walk(self.root):
			# 	print("ok", dirpath, dirnames)
			# 	if task_name in dirnames:
			# 		base_dirs.append(Path(dirpath) / task_name)
			# 	# ✅ stop early if found enough matches
			# 	if len(base_dirs) >= 3:
			# 		break
			dirnames = os.listdir(self.root)
			# print(task_name)
			if task_name in dirnames:
				base_dirs.append(Path(self.root) / task_name)
				# print(base_dirs)

		if not base_dirs:
			logger.warning("Task directory not found: %s", task_name)
			return []

		for base in base_dirs:
			try:
				with os.scandir(base) as it:
					top_files, top_dirs = [], []
					for entry in it:
						if entry.is_file():
							top_files.append(Path(entry.path))
						elif entry.is_dir():
							top_dirs.append(Path(entry.path))
					files.extend(top_files)
			except Exception:
				logger.exception("Failed to scan base dir: %s", base)
				continue

			# Shuffle and sample directories
			random.shuffle(top_dirs)
			print(top_dirs[0])
			print(len(top_dirs))
			dirs_to_scan = top_dirs

			# Distribute sampling across different top-level entries (dirs_to_scan).
			# If the caller requested a specific desired_count and it's <= number
			# of entries, prefer one file per directory across a sampled subset
			# of directories to maximize diversity.
			if dirs_to_scan:
				if desired_count is not None and desired_count <= len(dirs_to_scan):
					# Choose a subset of directories equal to desired_count and
					# set per_dir_cap=1 so we take at most one file per selected dir.
					dirs_to_scan = random.sample(dirs_to_scan, desired_count)
					per_dir_cap = 1
				else:
					per_dir_cap = 1
			else:
				per_dir_cap = MAX_CANDIDATES

			logger.debug("Sampling up to %d files per entry across %d entries", per_dir_cap, len(dirs_to_scan))

			# First pass: gather up to per_dir_cap files from each entry
			print(dirs_to_scan)
			for d in dirs_to_scan:
				collected = 0
				for root, _, filenames in os.walk(d):
					for fn in filenames:
						p = Path(root) / fn
						files.append(p)
						collected += 1
						if collected >= per_dir_cap or len(files) >= MAX_CANDIDATES:
							break
					if collected >= per_dir_cap or len(files) >= MAX_CANDIDATES:
						break
				if len(files) >= MAX_CANDIDATES:
					logger.debug("Reached candidate cap %d during distributed sampling", MAX_CANDIDATES)
					break

			# Second pass: if still short, fill remaining by scanning entries again (no per-dir cap)
			print('Len Files: ', len(files))
			if desired_count and desired_count > len(dirs_to_scan) and len(files) < MAX_CANDIDATES:
				print('Too Few files, second pass')
				for d in dirs_to_scan:
					for root, _, filenames in os.walk(d):
						for fn in filenames:
							p = Path(root) / fn
							files.append(p)
							if len(files) >= MAX_CANDIDATES:
								break
						if len(files) >= MAX_CANDIDATES:
							break
					if len(files) >= MAX_CANDIDATES:
						break

		# fallback: if still empty
		if not files:
			logger.debug("Fallback full scan under task_dir=%s", task_dir)
			if task_dir.exists():
				for p in task_dir.rglob("*"):
					if p.is_file():
						files.append(p)
						if len(files) >= MAX_CANDIDATES:
							break

		unique_files = list(dict.fromkeys(map(Path.resolve, files)))
		logger.debug("Gathered %d files (unique=%d) for %s", len(files), len(unique_files), task_name)
		return unique_files

	def build_eval_set(
		self,
		task_name: str,
		num_samples: int,
		out_path: Optional[str] = None,
		seed: Optional[int] = None,
	) -> Dict[str, List[str]]:
		"""Build a deduplicated sampled eval set for `task_name` and save it.

		Procedure:
		1. Gather all file paths under ``root/task_name`` into an array.
		2. Sample ``num_samples`` paths without replacement from that array.
		3. Save a JSON file of the form: {"<task_name>": ["<qa_path", ...]}.

		Args:
			task_name: name of the task subfolder under root.
			num_samples: desired number of QA files in the eval set.
			out_path: optional explicit path to save the mapping JSON. If not
					  provided, defaults to ``root/eval_sets/{task_name}_eval.json``.
			seed: optional random seed for reproducible sampling.

		Returns:
			The mapping dict that was saved.
		"""
		# all_files = self._gather_files_for_task(task_name)
		# Pass desired sample count to the gatherer so it can bias sampling for diversity
		
		all_files = self._gather_files_for_task(task_name, desired_count=num_samples)
		if not all_files:
			logger.warning("No files found for task %s; returning empty mapping", task_name)
			mapping: Dict[str, List[str]] = {task_name: []}
			# still save an empty mapping
			out_path = out_path or str(self.out / "eval_sets" / f"{task_name}_eval.json")
			os.makedirs(os.path.dirname(out_path), exist_ok=True)
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(mapping, f, indent=2, ensure_ascii=False)
			return mapping

		if seed is not None:
			random.seed(seed)

		n = min(num_samples, len(all_files))
		sampled = random.sample(all_files, n)

		# Store as string paths (absolute) to be unambiguous across runs
		sampled_strs = [str(p) for p in sampled]

		mapping = { task_name: sampled_strs }

		# Default out path
		if out_path is None:
			out_dir = self.out / "eval_sets"
			out_dir.mkdir(parents=True, exist_ok=True)
			out_path = str(out_dir / f"{task_name}_eval.json")

		# Save mapping
		try:
			os.makedirs(os.path.dirname(out_path), exist_ok=True)
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(mapping, f, indent=2, ensure_ascii=False)
			logger.info("Saved eval mapping for task %s to %s", task_name, out_path)
		except Exception:
			logger.exception("Failed to save eval mapping to %s", out_path)

		return mapping

	def load_eval_mapping(self, mapping_path: str) -> Dict[str, List[str]]:
		"""Load the saved mapping JSON and return the raw mapping.

		Args:
			mapping_path: path to the mapping JSON file created by
						  ``build_eval_set``.

		Returns:
			A dict mapping task_name to list of QA file paths.
		"""
		with open(mapping_path, "r", encoding="utf-8") as f:
			mapping = json.load(f)
		return mapping

	def load_eval_qa_list(self, mapping_path: Optional[str] = None, task_name: Optional[str] = None) -> List[Any]:
		"""Load QA JSON files referenced by a mapping and return list of QA objects.

		You can either pass a `mapping_path` (the JSON file that maps task to QA
		file paths) or pass `task_name` to load the default mapping file at
		``root/eval_sets/{task_name}_eval.json``.

		Args:
			mapping_path: optional explicit mapping JSON path.
			task_name: optional task name to use the default mapping location.

		Returns:
			A list where each element is the JSON-loaded content of a QA file.
		"""
		if mapping_path is None:
			if task_name is None:
				raise ValueError("Either mapping_path or task_name must be provided")
			mapping_path = str(self.out / "eval_sets" / f"{task_name}_eval.json")

		mapping = self.load_eval_mapping(mapping_path)

		# If task_name provided, use only that list; otherwise, load all tasks
		qa_paths: List[str] = []
		if task_name is not None and task_name in mapping:
			qa_paths = mapping[task_name]
		else:
			# flatten all lists
			for v in mapping.values():
				qa_paths.extend(v)

		qa_list: List[Any] = []
		for p in qa_paths:
			try:
				with open(p, "r", encoding="utf-8") as f:
					qa = json.load(f)
				qa_list.append(qa)
			except Exception:
				logger.exception("Failed to load QA file: %s", p)

		return qa_list


if __name__ == "__main__":
	# Quick demo when run as a script (not executed on import)
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("root", help="QA root directory")
	parser.add_argument("task", help="Task name under root to build eval set for")
	parser.add_argument("out", help="Output mapping file path")
	parser.add_argument("--n", type=int, default=20, help="Number of samples to pick")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	args = parser.parse_args()

	loader = QALoader(args.root, args.out)
	mapping = loader.build_eval_set(args.task, args.n, seed=args.seed)
	print("Saved mapping:", mapping)

