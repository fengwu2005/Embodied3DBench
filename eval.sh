#!/usr/bin/env bash
# Launcher for evaluator.constructors.eval_executor
# Defaults are taken from .vscode/launch.json; can be overridden via flags or environment variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"

# Defaults (from .vscode/launch.json)
ROOT_DEFAULT='./data/embodied3dbench'
TASK_DEFAULT='2d_oriented_bbox_qa_detect_category'
# '2d_oriented_bbox_qa_detect_all'
MODEL_CONFIG_DEFAULT='./configs/model_config.example.yaml'
MODEL_NAME_DEFAULT='openai-compatible'
#MODEL_NAME_DEFAULT="internvl3-38b"
#MODEL_NAME_DEFAULT="tanyuan-qwen2-5-vl-7b-local"
# MODEL_NAME_DEFAULT='gemini-2.5-pro'
# MODEL_NAME_DEFAULT='qwen2-5-vl-7b-local'
# MODEL_NAME_DEFAULT='internvl3-5-8b'
# MODEL_NAME_DEFAULT='doubao-seed-1-6-thinking'
# MODEL_NAME_DEFAULT='claude-sonnet-4'
LOADER_OUT_DEFAULT='./sampled_evalset'
OUT_DEFAULT='./results_qwen2vl7b_2dbbox_1216'
N_DEFAULT=0
SEED_DEFAULT=42

# Allow environment overrides
ROOT="${ROOT:-$ROOT_DEFAULT}"
TASK="${TASK:-$TASK_DEFAULT}"
MODEL_CONFIG="${MODEL_CONFIG:-$MODEL_CONFIG_DEFAULT}"
MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_DEFAULT}"
LOADER_OUT="${LOADER_OUT:-$LOADER_OUT_DEFAULT}"
OUT="${OUT:-$OUT_DEFAULT}"
N="${N:-$N_DEFAULT}"
SEED="${SEED:-$SEED_DEFAULT}"

usage(){
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra args passed to python module]

Options (overrides defaults / env vars):
  -R, --root PATH             Root path for pipeline outputs (default: ${ROOT_DEFAULT})
  -T, --task NAME             Task name (default: ${TASK_DEFAULT})
  -C, --model-config PATH     Path to model_config.yaml (default: ${MODEL_CONFIG_DEFAULT})
  -M, --model-name NAME       Model name (default: ${MODEL_NAME_DEFAULT})
  -L, --loader-out PATH       Loader out dir (default: ${LOADER_OUT_DEFAULT})
  -O, --out PATH              Output dir for results (default: ${OUT_DEFAULT})
  -N, --n NUM                 Number of examples to run (default: ${N_DEFAULT})
  -S, --seed NUM              Random seed (default: ${SEED_DEFAULT})
  -h, --help                  Show this help
  
Examples:
  # use defaults
  ./eval.sh

  # override some options
  ./eval.sh --root "/tmp/outputs" --task single_func_vector_2d --model-name qwen2-5-vl-7b -N 50

EOF
}

# Parse args: support short flags and long --key=value or --key value
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -R|--root)
      ROOT="$2"; shift 2;;
    --root=*) ROOT="${1#*=}"; shift 1;;
    -T|--task)
      TASK="$2"; shift 2;;
    --task=*) TASK="${1#*=}"; shift 1;;
    -C|--model-config)
      MODEL_CONFIG="$2"; shift 2;;
    --model-config=*) MODEL_CONFIG="${1#*=}"; shift 1;;
    -M|--model-name)
      MODEL_NAME="$2"; shift 2;;
    --model-name=*) MODEL_NAME="${1#*=}"; shift 1;;
    -L|--loader-out)
      LOADER_OUT="$2"; shift 2;;
    --loader-out=*) LOADER_OUT="${1#*=}"; shift 1;;
    -O|--out)
      OUT="$2"; shift 2;;
    --out=*) OUT="${1#*=}"; shift 1;;
    -N|--n)
      N="$2"; shift 2;;
    --n=*) N="${1#*=}"; shift 1;;
    -S|--seed)
      SEED="$2"; shift 2;;
    --seed=*) SEED="${1#*=}"; shift 1;;
    -h|--help)
      usage; exit 0;;
    --)
      shift; POSITIONAL+=("$@"); break;;
    --*=*) # unrecognized long option with =
      echo "Unknown option: $1"; usage; exit 1;;
    -*)
      echo "Unknown option: $1"; usage; exit 1;;
    *)
      POSITIONAL+=("$1"); shift;;
  esac
done

# Rebuild positional params
set -- "${POSITIONAL[@]}"

# Prepare environment: default PYTHONPATH to include evaluator (matches launch.json env)
export PYTHONPATH="${PYTHONPATH:-${WORKSPACE_DIR}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# echo $ROOT

CMD=(python -m constructors.eval_executor_paralle
  --root "$ROOT"
  --task "$TASK"
  --model-config "$MODEL_CONFIG"
  --model-name "$MODEL_NAME"
  --loader-out "$LOADER_OUT"
  --out "$OUT"
  --n "$N"
  --seed "$SEED"
)

# Append any extra positional args provided after --
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  CMD+=("${POSITIONAL[@]}")
fi

echo "Running evaluator with the following settings:"
echo "  ROOT          = $ROOT"
echo "  TASK          = $TASK"
echo "  MODEL_CONFIG  = $MODEL_CONFIG"
echo "  MODEL_NAME    = $MODEL_NAME"
echo "  LOADER_OUT    = $LOADER_OUT"
echo "  OUT           = $OUT"
echo "  N             = $N"
echo "  SEED          = $SEED"
echo "  PYTHONPATH    = $PYTHONPATH"
echo
echo "Invoking: ${CMD[*]}"

exec "${CMD[@]}"
