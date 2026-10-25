#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
cd "${ROOT}"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/eval.sh <HF-model-ID-or-checkpoint-dir> <num-loops> [evaluation.yaml]" >&2
  exit 2
fi
PYTHON="${LOOPQUANT_PYTHON:-python}"
model_path="$1"
num_loops="$2"
eval_config="${3:-${LOOPQUANT_EVAL_CONFIG:-${ROOT}/configs/evaluation_generative.yaml}}"
if [[ ! -f "${eval_config}" ]]; then
  echo "config not found: ${eval_config}" >&2
  exit 2
fi
model_key="$(basename "${model_path%/}")"
if [[ -f "${model_path}/checkpoint_manifest.json" ]]; then
  model_key="$(basename "$(dirname "${model_path}")")"
fi
output_dir="${LOOPQUANT_EVAL_OUTPUT_DIR:-${ROOT}/outputs/${model_key}_evaluation}"
mkdir -p "${output_dir}"
echo "[Evaluation] START model=${model_path}" | tee -a "${output_dir}/launcher.log"
backend_args=()
python_args=()
if [[ "${USE_VLLM:-1}" == 1 ]]; then
  backend_args+=(--vllm)
  python_args+=(-s)
fi
set +e
"${PYTHON}" "${python_args[@]}" -m loopquant.cli evaluate \
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
