#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
cd "${ROOT}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval.sh <experiment-name> [--model HF-ID-or-directory --num-loops N] [--eval-config FILE]" >&2
  exit 2
fi
source "${ROOT}/scripts/_launch_common.sh"
PYTHON="${LOOPQUANT_PYTHON:-python}"
run_name="$1"
shift
validate_run_name "${run_name}"
model_path=""
num_loops=""
eval_config="${LOOPQUANT_EVAL_CONFIG:-${ROOT}/configs/evaluation_generative.yaml}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--num-loops|--eval-config)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      case "$1" in
        --model) model_path="$2" ;;
        --num-loops) num_loops="$2" ;;
        --eval-config) eval_config="$2" ;;
      esac
      shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! -f "${eval_config}" ]]; then
  echo "config not found: ${eval_config}" >&2
  exit 2
fi
run_dir="${ROOT}/outputs/${run_name}"
if [[ -z "${model_path}" ]]; then
  if [[ -n "${num_loops}" ]]; then
    echo "--num-loops is only needed with --model; checkpoints use their saved loop count" >&2
    exit 2
  fi
  model_path="${run_dir}/checkpoint"
  manifest="${model_path}/checkpoint_manifest.json"
  if [[ ! -f "${manifest}" ]]; then
    echo "checkpoint manifest not found: ${manifest}" >&2
    exit 2
  fi
  num_loops="$(PYTHONIOENCODING=utf-8 "${PYTHON}" -s - "${manifest}" <<'PY'
import json, pathlib, sys
try:
    loops = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["num_loops"]
    if type(loops) is not int or loops <= 0:
        raise ValueError("num_loops must be a positive integer")
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"invalid checkpoint manifest: {error}")
print(loops)
PY
)"
  num_loops="${num_loops//$'\r'/}"
elif [[ ! "${num_loops}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--model requires --num-loops with a positive integer" >&2
  exit 2
fi
backend_args=()
case "${USE_VLLM:-1}" in
  1) backend="vllm"; backend_args+=(--vllm) ;;
  0) backend="hf" ;;
  *) echo "USE_VLLM must be 0 or 1" >&2; exit 2 ;;
esac
output_dir="${run_dir}/evaluation/${backend}"
mkdir -p "${output_dir}"
echo "[Evaluation] START model=${model_path} backend=${backend}" | tee -a "${output_dir}/launcher.log"
set +e
"${PYTHON}" -s -m loopquant.cli evaluate \
  --model "${model_path}" \
  --num-loops "${num_loops}" \
  --eval-config "${eval_config}" \
  --output-dir "${output_dir}" \
  --wikitext2 --c4 --zero-shot --generative \
  "${backend_args[@]}" \
  2>&1 | tee -a "${output_dir}/launcher.log"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[Evaluation] FAILED status=${status}" | tee -a "${output_dir}/launcher.log"
  exit "${status}"
fi
echo "[Evaluation] DONE output=${output_dir}" | tee -a "${output_dir}/launcher.log"
echo "[Evaluation] outputs subdirectory: ${run_name}" | tee -a "${output_dir}/launcher.log"
