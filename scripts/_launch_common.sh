#!/usr/bin/env bash
# Sourced by the launchers after ROOT/PYTHONPATH and the repository cwd are set.
validate_run_name() {
  case "$1" in
    ""|"."|".."|*/*|*\\*|*$'\n'*|*$'\r'*)
      echo "experiment name/tag must be a single directory name: $1" >&2
      return 2 ;;
  esac
}

load_run_config() {
  config="$1"
  eval_config="$2"
  PYTHON="${LOOPQUANT_PYTHON:-python}"
  for path in "${config}" "${eval_config}"; do
    if [[ ! -f "${path}" ]]; then
      echo "config not found: ${path}" >&2
      return 2
    fi
  done
  local config_values
  config_values="$(PYTHONIOENCODING=utf-8 "${PYTHON}" -s - "${config}" <<'PY'
import pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(config.get("model_path"), str) or not config["model_path"].strip():
    raise ValueError("model_path must be a nonempty path in the run YAML")
print(config["model_path"])
PY
)" || return $?
  # Native Windows Python emits CRLF when the launchers are checked via MSYS.
  config_values="${config_values//$'\r'/}"
  model_path="${config_values}"
  local model_root
  if [[ -n "${LOOPQUANT_MODEL_ROOT:-}" ]]; then
    model_root="${LOOPQUANT_MODEL_ROOT}"
  elif [[ -d "${ROOT}/../loopquant/models" ]]; then
    model_root="${ROOT}/../loopquant"
  else
    model_root="${ROOT}/.."
  fi
  if [[ "${model_path}" != /* && ! "${model_path}" =~ ^[A-Za-z]:[/\\] ]]; then
    if [[ -d "${model_root}/${model_path}" ]]; then
      model_path="${model_root}/${model_path}"
    elif [[ -d "${model_path}" ]]; then
      : # An existing path relative to the repository is already usable.
    elif [[ "${model_path}" == models/* || "${model_path}" == ./* || "${model_path}" == ../* ]]; then
      model_path="${model_root}/${model_path}"
    fi
    # Otherwise preserve the source as a HF model ID for the Python loader.
  fi
}
