#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
cd "${ROOT}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/quantize.sh configs/<run>.yaml [--tag TAG] [--eval-config FILE]" >&2
  exit 2
fi
source "${ROOT}/scripts/_launch_common.sh"
config="$1"
shift
tag=""
eval_config="${LOOPQUANT_EVAL_CONFIG:-${ROOT}/configs/evaluation_generative.yaml}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag|--eval-config)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "missing value for $1" >&2
        exit 2
      fi
      if [[ "$1" == --tag ]]; then tag="$2"; else eval_config="$2"; fi
      shift 2 ;;
    --*) echo "unknown argument: $1" >&2; exit 2 ;;
    *)
      # Preserve the previous optional positional evaluation YAML.
      eval_config="$1"
      shift ;;
  esac
done
run_name="$(basename "${config}")"
run_name="${run_name%.*}"
validate_run_name "${run_name}"
if [[ -n "${tag}" ]]; then
  validate_run_name "${tag}"
  run_name+="_${tag}"
fi
load_run_config "${config}" "${eval_config}"
output_dir="${ROOT}/outputs/${run_name}"
mkdir -p "${output_dir}"
echo "[LoopQuant] START config=${config} model=${model_path}" | tee -a "${output_dir}/launcher.log"
set +e
"${PYTHON}" -s -m loopquant.cli quantize \
  --model "${model_path}" \
  --config "${config}" \
  --output-dir "${output_dir}" \
  --eval-config "${eval_config}" \
  --save --wikitext2 --c4 --zero-shot \
  2>&1 | tee -a "${output_dir}/launcher.log"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "[LoopQuant] FAILED config=${config} status=${status}" | tee -a "${output_dir}/launcher.log"
  exit "${status}"
fi
echo "[LoopQuant] DONE config=${config}" | tee -a "${output_dir}/launcher.log"
echo "[LoopQuant] outputs subdirectory: ${run_name}" | tee -a "${output_dir}/launcher.log"
