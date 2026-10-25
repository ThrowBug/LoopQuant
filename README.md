# LoopQuant

Loop-aware post-training quantization for Ouro, with independent checkpoint and
model-quality evaluation routes. Current model adapters validate Ouro-1.4B and
Ouro-2.6B. Nanbeige/Huginn YAMLs remain historical configurations; their adapters
are not implemented.

## Setup

```bash
conda create -n loopquant python=3.12
conda activate loopquant
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Original models can be local HF directories or HF model IDs. Quantization uses
WikiText-2 training windows (default 128 x 2048). Evaluation datasets and MBPP's
metric scripts are downloaded on first use.

## Two commands

- `quantize`: load original weights, run CLDC -> ILPA -> LCTC, optionally save,
  then optionally evaluate the in-memory result.
- `evaluate`: load original HF weights **or a saved LoopQuant checkpoint**, then
  evaluate. It never recalibrates or calls the quantization pipeline.

`--save`, `--wikitext2`, `--c4`, `--zero-shot`, and `--generative` are `store_true`
flags and default to false. `--save` belongs only to `quantize`. Evaluation YAMLs
supply protocol parameters, not execution switches. Removed execution switches in older YAMLs are rejected to avoid silently ignored settings. An `evaluate`
command with no selected suite is rejected before model loading.

### Shell launchers

`quantize.sh` explicitly enables saving, WikiText/C4 PPL and zero-shot evaluation;
generative evaluation is disabled in this launcher. `eval.sh` enables all four
evaluation suites. Edit their flag lists to select a different combination.
`eval.sh` defaults to `USE_VLLM=1`, so it skips PPL. Set `USE_VLLM=0` to use HF
and run all selected suites, including PPL. Direct CLI calls still require an
explicit `--vllm` flag to select vLLM.
Both launchers default to `TOKENIZERS_PARALLELISM=false` and
`HF_ALLOW_CODE_EVAL=1`; existing environment values override these defaults.

```bash
# Quantize -> save -> PPL/zero-shot. Reads model_path/output_dir from the run YAML.
conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/quantize.sh configs/ouro_1.4b_w4a4.yaml

# vLLM: original HF model (replace the ID with a local HF directory if needed).
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  LOOPQUANT_EVAL_OUTPUT_DIR=outputs/ouro_1.4b_bf16_vllm_eval \
  bash scripts/eval.sh ByteDance/Ouro-1.4B 4

# vLLM: saved fake-quantized model.
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  LOOPQUANT_EVAL_OUTPUT_DIR=outputs/ouro_1.4b_w4a4_vllm_eval \
  bash scripts/eval.sh outputs/ouro_1.4b_w4a4/loopquant 4

# HF (without vLLM): original HF model, including PPL evaluation.
conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 \
  LOOPQUANT_EVAL_OUTPUT_DIR=outputs/ouro_1.4b_bf16_hf_eval \
  bash scripts/eval.sh ByteDance/Ouro-1.4B 4

# HF (without vLLM): saved fake-quantized model, including PPL evaluation.
conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 \
  LOOPQUANT_EVAL_OUTPUT_DIR=outputs/ouro_1.4b_w4a4_hf_eval \
  bash scripts/eval.sh outputs/ouro_1.4b_w4a4/loopquant 4
```

`eval.sh` requires the loop count as its second argument. An optional third
argument selects another evaluation YAML (`quantize.sh`: optional second argument). The default is
`configs/evaluation_generative.yaml`; `LOOPQUANT_EVAL_CONFIG` overrides that default.
`LOOPQUANT_PYTHON` optionally selects a Python executable, otherwise scripts use
`python` from the active environment. Spaces in paths are preserved.

For `quantize.sh`, relative model paths such as `models/Ouro-1.4B` are resolved
against `LOOPQUANT_MODEL_ROOT` (default: repository parent, with the historical
`../loopquant/models` fallback). Existing directories under that root take priority,
followed by existing repository-relative directories. HF IDs such as
`ByteDance/Ouro-1.4B` pass through unchanged when no matching local directory exists.
The `models/`, `./`, and `../` prefixes explicitly select local paths even when
the directory is missing; absolute paths also pass through unchanged.
`eval.sh` takes a direct model ID/path, not a
quantization YAML. Relative paths are relative to the repository. Evaluation output
defaults to `outputs/<model-name>_evaluation` (the parent run name for a snapshot);
`LOOPQUANT_EVAL_OUTPUT_DIR` overrides it. Use separate output directories per run.

### Direct CLI examples

```bash
# Only quantize and save. No evaluation/scorer dependencies are loaded.
python -m loopquant.cli quantize \
  --model ByteDance/Ouro-1.4B --config configs/ouro_1.4b_w4a4.yaml \
  --output-dir outputs/ouro_1.4b_w4a4 --save

# Quantize, save successfully, then evaluate all suites.
HF_ALLOW_CODE_EVAL=1 python -m loopquant.cli quantize \
  --model /models/Ouro-1.4B --config configs/ouro_1.4b_w4a4.yaml \
  --output-dir outputs/ouro_1.4b_w4a4_full --save \
  --wikitext2 --c4 --zero-shot --generative

# Evaluate a saved quantized checkpoint; only the selected suites run.
python -m loopquant.cli evaluate \
  --model outputs/ouro_1.4b_w4a4/loopquant --num-loops 4 \
  --output-dir outputs/ouro_1.4b_w4a4_reeval --wikitext2 --zero-shot

# Full-precision baseline through the same command.
HF_ALLOW_CODE_EVAL=1 python -m loopquant.cli evaluate \
  --model ByteDance/Ouro-1.4B --num-loops 4 --output-dir outputs/ouro_1.4b_bf16 --generative
```

`--eval-config` selects protocol parameters for either command. Full-precision
loading defaults to BF16; `evaluate --num-loops` is required for all sources, and
quantization YAMLs must explicitly provide `num_loops`. Saved snapshots preserve
tensor dtypes and require the supplied loop count to match; dtype overrides are
rejected for snapshots. Original models accept `--dtype float32`.
The old `evaluate --run-dir` replay and `evaluate-bf16` commands are removed.

### Optional vLLM evaluation

`evaluate --vllm` runs only zero-shot and generative tasks. Selected WikiText/C4
PPL suites are skipped explicitly in the log and manifest. A PPL-only vLLM command
is rejected. The default HF route and the quantization command are unchanged.

Create a **separate Linux x86_64/CUDA Conda environment**, then from the repository root install:

```bash
conda create -n loopquant-vllm python=3.12 -y
conda activate loopquant-vllm
python -I -m pip --isolated install --no-user \
  -i https://mirrors.aliyun.com/pypi/simple/ -r requirements-vllm.txt
python -I -m pip check
```

`requirements-vllm.txt` locks vLLM 0.11.2, Torch 2.9.0 (CUDA 12.8),
Transformers 4.57.1 and lm-eval 0.4.11, including their transitive dependencies.
The standard PyPI wheels are installed through Aliyun; no GitHub wheel URL is
needed. The target is CPython 3.12, Linux x86_64 and glibc >= 2.28. Installation
does not require root. Use a fresh environment: reinstalling into one with missing
package metadata/RECORD files is not a supported repair procedure.
`-I` hides user-site packages and `PYTHONPATH`; `--isolated --no-user` prevents
user pip settings from redirecting the installation. Verify the installation:

```bash
CUDA_VISIBLE_DEVICES=0 python -I - <<'PY'
from importlib.metadata import version
import torch

print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("vLLM:", version("vllm"))
torch.cuda.init()
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA computation:", (torch.ones(1, device="cuda") + 1).item())
import vllm._C
PY
```

The CUDA 12.8 route relies on NVIDIA's [CUDA 12.x minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
on older drivers such as R550 on A800. It is a candidate installation, not a guarantee that the full
engine runs: newer driver features and PTX/JIT compilation can still fail.
Validate an actual vLLM run or the integration smoke test below on the server.
The lock keeps CUDA runtime dependencies on CUDA 12.x. Dependency resolution
does not validate GPU kernels on the server.

Original HF weights use vLLM's [native Ouro model](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/ouro.py).
LoopQuant snapshots select the custom worker plugin installed by the editable
`-e .` entry. Both routes use vLLM's V1 engine; unset `VLLM_USE_V1=0` if present.
Keep `loopquant` for quantization/HF evaluation and `loopquant-vllm` for vLLM
evaluation. Their Transformers/lm-eval pins differ; install only the appropriate
requirements file in each environment. The shared evaluation code supports both
harness versions. The model's custom HF source must be importable; saved
LoopQuant directories already include it.

```bash
# Run from the LoopQuant repository root in the vLLM environment.
conda activate loopquant-vllm

# Original BF16 model: zero-shot + all configured generative tasks, including MBPP.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python -s -m loopquant.cli evaluate --vllm \
  --model ByteDance/Ouro-1.4B --num-loops 4 \
  --eval-config configs/evaluation_generative.yaml \
  --output-dir outputs/ouro_bf16_vllm --zero-shot --generative

# Saved LoopQuant model: load the existing snapshot, then evaluate through vLLM.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python -s -m loopquant.cli evaluate --vllm \
  --model outputs/ouro_1.4b_w4a4/loopquant --num-loops 4 \
  --eval-config configs/evaluation_generative.yaml \
  --output-dir outputs/ouro_1.4b_w4a4_vllm_eval --zero-shot --generative

# Equivalent launcher for a saved snapshot; USE_VLLM defaults to 1.
USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh outputs/ouro_1.4b_w4a4/loopquant 4

# The launcher also accepts an original HF model ID or local model directory.
USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ByteDance/Ouro-1.4B 4

# Use the original HF environment/backend (including PPL).
conda activate loopquant
USE_VLLM=0 \
  bash scripts/eval.sh outputs/ouro_1.4b_w4a4/loopquant 4
```

`--vllm`/`USE_VLLM=1` skips WikiText/C4 PPL; it does not change the generation
task list or few-shot settings in `evaluation_generative.yaml`. To run only the
zero-shot suite, omit `--generative` in the direct CLI command; MBPP execution
permission is then unnecessary. The loop count must match a saved snapshot.

The snapshot bridge reconstructs the saved HF decoder, CLDC transforms and LCTC
tables, preserving unfused projections and exact tensor dtypes (including FP32
CGB parameters). Only attention execution is replaced by vLLM paged attention;
each physical layer and recurrent loop has its own cache entry. The existing
manifest/safetensors format is reused. No packed INT4 kernel is introduced.

The integration is single-GPU, eager, with torch.compile, CUDA Graph and prefix caching disabled.
The vLLM launcher uses `python -s` to keep packages in `~/.local` out of the
runtime import path. Direct CLI examples use the same option. vLLM's registry
starts a separate Python interpreter without copying `-s`, so backend initialization
also temporarily sets `PYTHONNOUSERSITE=1` for its child processes. The previous
value is restored in the calling process when initialization finishes or fails;
no shell export is needed.
Engine seed, context budget, task batch sizes and truncation policy follow the
evaluation protocol. vLLM always uses its KV cache, so `generative.use_cache: false`
is rejected rather than silently ignored. `CUDA_VISIBLE_DEVICES` selects the GPU;
the logical device must be `cuda:0` (or `auto`). Engine arguments, skipped suites,
versions and per-request generation receipts are saved. Preflight and model-load
failures also produce a failed manifest.

CPU reference tests cover tensor restoration, RoPE, recurrent prefill and
incremental decode with W4A4/mixed A4/A8 and LCTC. They do **not** exercise the real
CUDA paged-attention implementation. Run the optional integration smoke test in
the target vLLM environment before relying on benchmark results; actual Ouro
GPU accuracy/performance remains an acceptance check, not a claim of these tests.

```bash
python -I -m pip --isolated install --no-user -i https://mirrors.aliyun.com/pypi/simple/ pytest
LOOPQUANT_VLLM_TEST_MODEL=/path/to/loopquant \
  python -s -m pytest tests/test_vllm_gpu.py -q

# Repeat for the original HF model and saved snapshots. Optional HF score
# comparison loads another model copy on the GPU and checks numerical agreement.
LOOPQUANT_VLLM_TEST_MODEL=ByteDance/Ouro-1.4B LOOPQUANT_VLLM_COMPARE_HF=1 \
  python -s -m pytest tests/test_vllm_gpu.py -q
```

## Checkpoint format

`--save` writes `output-dir/loopquant` **before any evaluation task/scorer is
loaded**. Missing MBPP execution settings or failed scoring therefore cannot
prevent a completed model snapshot being produced. Save errors stop the route
before evaluation. Existing checkpoint destinations are never overwritten.
Temporary staging directories are published only after the snapshot is complete.

A versioned `checkpoint_manifest.json` contains the structure, exact loop-bit
allocation, module flags, parameter aliases, shard index, provenance and SHA-256
checksums. The directory also contains:

```text
config.json
model/config definition files for custom HF classes
[tokenizer files]
generation_config.json
weights-00001.safetensors
[additional weight shards]
```

The snapshot stores **floating-point fake-quantized weights**, not packed INT4.
It includes changed embeddings/LM head, rotations, CLDC scales/permutations,
clipping tables, and the LCTC CGB parameters/buffers. CGB FP32 dtypes are retained.
Restoration builds an empty model skeleton, recreates known wrappers, assigns
saved tensors, restores aliases and registers runtime hooks. It does not rotate
weights a second time, run GPTQ, load a teacher, or access calibration data.
Original model definitions are copied so loading does not depend on the original
weight directory. The LoopQuant package itself must remain installed/available.

Format v1 supports the current Ouro architecture plus LoopQuant wrappers. An
arbitrary injected module needs its own reconstruction support before it can be
saved in this format; prepared HF-compatible models can still be evaluated via
the Python API. Unsupported/corrupt snapshot formats raise errors, not a silent
fallback to full-precision loading.

Teacher loading no longer resets the student's global activation allocation.
Pipeline completion and checkpoint saving check that the actual context matches
the selected allocation; a restored model re-establishes its saved context.
The context is still global: use separate processes for simultaneous models, or
explicitly manage it when switching prepared models within one process.

## Evaluation protocol

The supplied protocol covers:

| Suite | Task | Few-shot |
| --- | --- | --- |
| Math generation | gsm8k_cot | 8 |
| Short-answer knowledge | triviaqa | 5 |
| Reasoning | bbh_cot_fewshot | Task defaults |
| Reading/numerical QA | drop | 1 |
| Python generation | mbpp | 3 |

The TriviaQA/DROP shot counts and output budgets are explicit experiment choices,
not a claim to reproduce published model scores. Few-shot examples are provided
or sampled by the harness. Base Ouro uses completion prompts, not a chat template.
The zero-shot flag retains the original ARC/HellaSwag/LAMBADA/PIQA/WinoGrande suite.
WikiText-2 and C4 perplexity are computed by this repository.

Generation preserves task-native stop strings, extraction and metrics. Greedy
sampling and per-task output budgets are explicit. The supplied generation batch
size is 8, with truncation permitted; use batch size 1 and disable truncation for
initial compatibility checks. `max_length` is a total prompt/output budget,
independent of the calibration window. With `allow_truncation: false`, prompts
that exceed the remaining budget fail instead of silently losing context.
`use_cache` is explicit, with no silent fallback. Set it false to diagnose Ouro
cache issues. Keep all settings identical between BF16 and quantized comparisons.

`limit` is a positive example count per harness subtask, or a window count for
PPL. Set a count for development or null for a full run; `full_eval` is derived
from this value in the results and is no longer an input parameter.
The harness and Evaluate versions are pinned; resolved task configs, dataset
fingerprints and versions are retained because remote artifacts can still change.

### MBPP environment

Use Linux/WSL and install `requirements.txt` in the model's Python environment.
Docker is not a dependency. Scoring executes generated Python on CPU; Conda only
isolates dependencies, so use a dedicated low-privilege account or isolated
execution environment. Native Windows code scoring is unsupported.

MBPP requires both `generative.confirm_run_unsafe_code: true` in the protocol and
`HF_ALLOW_CODE_EVAL=1` in the process environment. These switches allow execution;
they do not create isolation. Both shell launchers default the environment
variable to `1`; direct Python calls must set it explicitly. The scorer self-check
runs when evaluation tasks are loaded.

### Prepared-model Python API

```python
from loopquant.evaluation import run_evaluation
from loopquant.evaluation_config import load_evaluation_config

settings = load_evaluation_config("configs/evaluation_generative.yaml")
# model, tokenizer = your_loader_or_quantizer(...)
run_evaluation(model, tokenizer, settings, "outputs/my_method",
               suites=["generative"], device="cuda:0",
               run_metadata={"method": "my_method"})
```

The runner does not load weights, run PTQ or reset bit allocations. Custom model
loaders own their modules and runtime state. Models must preserve HF causal-LM
forward/generate semantics, or use an appropriate adapter.

### Results and tests

Each completed suite/task is saved immediately. `metrics.json` holds aggregate
results; `evaluation_manifest.json` records effective switches, protocol, model
source, versions and completion/failure state. Harness summaries retain subtask
and group scores. Sample JSONL retains prompts, responses, extracted answers and
scores. `generation_receipts.json` records batch lengths, cache policy and length
cap hits (not per-example stopping reasons). A failed evaluation preserves earlier
scores. Reruns do not automatically resume or reuse responses.

```bash
pip install pytest
python -m pytest tests -q
```

Offline tests use tiny HF recurrent models to check state/logit/generation identity
across checkpoint restoration in a fresh process, W4A4/mixed W4A6 allocation,
FP32 CGB tensors, corruption detection, all four save/evaluate combinations,
real HF/harness generation, and shell argument/exit handling. Full Ouro GPU
benchmarks and Linux MBPP execution must still be validated in the target setup.

## License

This work is released under the Apache 2.0 license.
