#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "${ROOT}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/analyze_activations.sh <experiment-name> (--model HF-ID-or-directory --num-loops N | --teacher-cache DIRECTORY) [analysis options]" >&2
  exit 2
fi
source "${ROOT}/scripts/_launch_common.sh"
run_name="$1"
shift
validate_run_name "${run_name}"
PYTHON="${LOOPQUANT_PYTHON:-python}"
output_dir="${ROOT}/outputs/${run_name}/activation_analysis"
mkdir -p "${output_dir}"
echo "[ActivationAnalysis] START experiment=${run_name}" | tee -a "${output_dir}/launcher.log"
set +e
"${PYTHON}" -s -m loopquant.activation_analysis "$@" --output-dir "${output_dir}" \
  2>&1 | tee -a "${output_dir}/launcher.log"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[ActivationAnalysis] FAILED status=${status}" | tee -a "${output_dir}/launcher.log"
  exit "${status}"
fi
echo "[ActivationAnalysis] DONE output=${output_dir}" | tee -a "${output_dir}/launcher.log"
echo "[ActivationAnalysis] outputs subdirectory: ${run_name}" | tee -a "${output_dir}/launcher.log"
