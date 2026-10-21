# Embodied3DBench Evaluation Toolkit

Evaluation code for **Embodied3DBench: Benchmarking Low-Level Embodied Spatial Intelligence of Vision Language Models** ([arXiv:2605.29074](https://arxiv.org/abs/2605.29074)).

Embodied3DBench evaluates whether vision-language models understand the low-level spatial structure needed for embodied interaction. The benchmark contains more than 21,000 QA pairs across six task categories and twelve 2D/3D subcategories.

## Tasks

| Group | Category | Representative outputs |
|---|---|---|
| Spatial structural understanding | Grounding | 2D points/boxes and 3D points/boxes |
| Spatial structural understanding | Spatial relation | relative position and metric distance |
| Spatial structural understanding | Multi-view correspondence | corresponding 2D/3D points |
| Interaction-oriented perception | Affordance | functional points and directions |
| Interaction-oriented perception | Grasp prediction | 2D/3D grasp points |
| Interaction-oriented perception | Trajectory prediction | 2D/3D manipulation trajectories |

The evaluator normalizes model responses, computes task-specific raw metrics, and converts them to benchmark scores. Supported raw metrics include 2D/3D IoU, L2 distance, absolute distance error, cosine similarity, accuracy, and Hausdorff distance.

## Repository layout

```text
Embodied3DBench/
├── adapters/                 # API and local-model adapters
├── configs/                  # Response schemas and model config example
├── constructors/             # QA loading and evaluation executors
├── metrics/                  # Task metrics and score aggregation
├── eval.sh                   # Single-task launcher
├── eval_all.py               # Multi-task launcher
└── score.py                  # Result-table utility
```

Benchmark data, model weights, predictions, and API credentials are not included in Git.

## Installation

```bash
git clone <repository-url>
cd Embodied3DBench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Local Qwen adapters require a CUDA-enabled PyTorch installation. The ESI-VLM adapter additionally requires the sibling ESI-VLA repository:

```bash
pip install -e ../ESI-VLA
```

## Data layout

Place the released benchmark under a local directory using one of the layouts recognized by `QALoader`:

```text
data/embodied3dbench/
└── outputs/
    ├── 2d_oriented_bbox_qa_detect_category/
    ├── correspondence_pairs_customized_3d/
    ├── grasp_point_2d/
    ├── traj_qa_prediction_3d/
    └── ...
```

Each task directory contains QA JSON files. Image paths inside each QA record must resolve on the evaluation machine. Keep public test annotations separate if the benchmark uses a hidden-test protocol.

## Configure a model

Copy `configs/model_config.example.yaml` to a local ignored file and set credentials through environment variables:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="..."
```

For local Qwen3-VL evaluation:

```bash
export QWEN3_VL_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
```

Never commit API keys or private service endpoints.

## Run evaluation

From the repository root:

```bash
./eval.sh \
  --root data/embodied3dbench \
  --task grasp_point_2d \
  --model-config configs/model_config.example.yaml \
  --model-name openai-compatible \
  --loader-out outputs/eval_sets \
  --out outputs/results/grasp_point_2d \
  --n 100
```

The evaluator samples QA files deterministically when a seed is supplied, sends the image/question to the selected adapter, parses its structured JSON answer, computes the task metric, and writes prediction records under the output directory.

`eval_all.py` contains the full task list used for batch evaluation. Review its model list, output path, sampling count, and worker count before launching a large run.

## Evaluate ESI-VLM

Install ESI-VLA, add a `qwen3-depth-encoder` entry to your local model configuration, and provide:

- `model.vlm.model_path` for Qwen3-VL;
- `model.depth_encoder.backbone_path` for Depth Anything V3;
- `extra.ckpt_path` for the trained ESI-VLM checkpoint.

The adapter is loaded lazily so API-only evaluation does not require ESI-VLA dependencies.

## Reproducibility

For a paper result, record the benchmark release/version, exact task manifest hashes, model revision, prompt/schema revision, random seed, reasoning setting, decoding parameters, number of evaluated samples, and evaluator commit.

## Citation

Citation metadata will be added for the public release.

## License

The evaluation code is released under the MIT License. Benchmark data and model outputs may have separate terms; consult the corresponding release pages before redistribution.
