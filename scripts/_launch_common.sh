#!/usr/bin/env bash
# Sourced by the launchers after ROOT/PYTHONPATH and the repository cwd are set.
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
  config_values="$(PYTHONIOENCODING=utf-8 "${PYTHON}" - "${config}" <<'PY'
import pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("model_path", "output_dir"):
    if not isinstance(config.get(key), str) or not config[key].strip():
        raise ValueError(f"{key} must be a nonempty path in the run YAML")
print(config["model_path"])
print(config["output_dir"])
PY
)" || return $?
  # Native Windows Python emits CRLF when the launchers are checked via MSYS.
  config_values="${config_values//$'\r'/}"
  local fields
  mapfile -t fields <<<"${config_values}"
  model_path="${fields[0]}"
  output_dir="${fields[1]}"
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
