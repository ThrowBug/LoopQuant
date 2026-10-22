#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/quantize.sh configs/<run>.yaml [evaluation.yaml]" >&2
  exit 2
fi
source "${ROOT}/scripts/_launch_common.sh"
load_run_config "$1" "${2:-${LOOPQUANT_EVAL_CONFIG:-${ROOT}/configs/evaluation_generative.yaml}}"
mkdir -p "${output_dir}"
echo "[LoopQuant] START config=${config} model=${model_path}" | tee -a "${output_dir}/launcher.log"
set +e
"${PYTHON}" -m loopquant.cli quantize \
  --model "${model_path}" \
  --config "${config}" \
  --output-dir "${output_dir}" \
  --eval-config "${eval_config}" \
  --save --wikitext2 --c4 --zero-shot --generative \
  2>&1 | tee -a "${output_dir}/launcher.log"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[LoopQuant] FAILED config=${config} status=${status}" | tee -a "${output_dir}/launcher.log"
  exit "${status}"
fi
echo "[LoopQuant] DONE config=${config}" | tee -a "${output_dir}/launcher.log"
