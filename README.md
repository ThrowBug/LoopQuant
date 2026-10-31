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

- `quantize`: load original weights, run the configured algorithm, optionally save,
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
# Quantize -> save -> PPL/zero-shot. An omitted/empty tag adds no suffix.
conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/quantize.sh configs/ouro_1.4b_w4a4.yaml --tag baseline

# vLLM: original HF model (replace the ID with a local HF directory if needed).
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_bf16 --model ByteDance/Ouro-1.4B --num-loops 4

# vLLM: saved fake-quantized model.
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4_baseline

# HF (without vLLM): original HF model, including PPL evaluation.
conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_bf16 --model ByteDance/Ouro-1.4B --num-loops 4

# HF (without vLLM): saved fake-quantized model, including PPL evaluation.
conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4_baseline
```

`quantize.sh` names the experiment after the config filename without its extension,
optionally followed by `_<tag>`. It overrides the YAML's legacy `output_dir` value.
`eval.sh` takes this experiment name, loads `outputs/<name>/checkpoint`, and reads
`num_loops` from its manifest. For an original BF16 model, supply `--model` and
`--num-loops`; the experiment name then only selects the result directory.
Names and tags must be single directory names, without path separators.

Both launchers accept `--eval-config FILE`. The default is
`configs/evaluation_generative.yaml`; `LOOPQUANT_EVAL_CONFIG` overrides that default.
`quantize.sh` also retains its previous optional positional evaluation YAML.
`LOOPQUANT_PYTHON` optionally selects a Python executable, otherwise scripts use
`python` from the active environment. Spaces in paths are preserved.

For `quantize.sh`, relative model paths such as `models/Ouro-1.4B` are resolved
against `LOOPQUANT_MODEL_ROOT` (default: repository parent, with the historical
`../loopquant/models` fallback). Existing directories under that root take priority,
followed by existing repository-relative directories. HF IDs such as
`ByteDance/Ouro-1.4B` pass through unchanged when no matching local directory exists.
The `models/`, `./`, and `../` prefixes explicitly select local paths even when
the directory is missing; absolute paths also pass through unchanged.
For `eval.sh --model`, relative model paths are relative to the repository.
Results follow a fixed layout; `LOOPQUANT_EVAL_OUTPUT_DIR` is no longer used:

```text
outputs/<config-stem>[_<tag>]/
  config.yaml
  quantization_summary.json
  <stage-summary>.json
  run.log
  launcher.log
  checkpoint/
    checkpoint_manifest.json
    ... model, tokenizer and tensor files ...
  evaluation/
    hf/                     # immediate quantization evaluation or standalone HF
      metrics.json
      evaluation_manifest.json
      zero_shot/...
      generative/<task>/...
    vllm/                   # standalone vLLM, PPL skipped
      metrics.json
      evaluation_manifest.json
      zero_shot/...
      generative/<task>/...
```

Only selected suites produce their result files. Standalone evaluation also writes
its protocol and logs inside the backend directory. BF16 experiments contain
`evaluation/` without a saved checkpoint or quantization files. HF and vLLM results
are separate; rerunning the same backend updates that backend's result files.
Quantization refuses to overwrite an existing checkpoint; select a new tag for a
new run. Existing output directories are not moved. The direct Python CLI still
accepts explicit model/checkpoint paths and evaluation output directories.

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
  --model outputs/ouro_1.4b_w4a4/checkpoint --num-loops 4 \
  --output-dir outputs/ouro_1.4b_w4a4/evaluation/hf --wikitext2 --zero-shot

# Full-precision baseline through the same command.
HF_ALLOW_CODE_EVAL=1 python -m loopquant.cli evaluate \
  --model ByteDance/Ouro-1.4B --num-loops 4 \
  --output-dir outputs/ouro_1.4b_bf16/evaluation/hf --generative
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
  --output-dir outputs/ouro_1.4b_bf16/evaluation/vllm --zero-shot --generative

# Saved LoopQuant model: load the existing snapshot, then evaluate through vLLM.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python -s -m loopquant.cli evaluate --vllm \
  --model outputs/ouro_1.4b_w4a4/checkpoint --num-loops 4 \
  --eval-config configs/evaluation_generative.yaml \
  --output-dir outputs/ouro_1.4b_w4a4/evaluation/vllm --zero-shot --generative

# Equivalent launcher for a saved snapshot; USE_VLLM defaults to 1.
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh ouro_1.4b_w4a4

# Supply an original HF model ID or local model directory explicitly.
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_bf16 --model ByteDance/Ouro-1.4B --num-loops 4

# Use the original HF environment/backend (including PPL).
conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh ouro_1.4b_w4a4
```

`--vllm`/`USE_VLLM=1` skips WikiText/C4 PPL; it does not change the generation
task list or few-shot settings in `evaluation_generative.yaml`. To run only the
zero-shot suite, omit `--generative` in the direct CLI command; MBPP execution
permission is then unnecessary. The loop count must match a saved snapshot.

The snapshot executor restores the existing manifest/safetensors without
re-quantizing weights. It reuses vLLM's native Ouro decoder control flow and paged
attention, with separate KV entries for each physical layer and recurrent loop.
Q/K/V and gate/up keep their independent CLDC transforms, activation quantizers
and LCTC parameters. Static per-loop execution views share the restored weights;
FP32 CGB tables and the original rotated RMSNorm dtype boundaries are retained.
Residual addition is rounded before normalization, as on the HF path. No packed
INT4 kernel is introduced, and quantization/HF evaluation code is unchanged.

Saved snapshots use `LOOPQUANT_VLLM_EXECUTION=cached` by default. It caches fixed
Hadamard/small orthogonal matrices on the device, sharing them across projections,
and pre-expands LCTC gain/bias after loading. Forward calls neither rebuild these
constants nor read the global quantization context. Matrix caches are per engine,
consume additional resident VRAM, and do not enter the checkpoint.

One optional environment setting selects more aggressive execution:

| Value | Snapshot execution |
| --- | --- |
| `cached` (default) | Reference arithmetic in eager mode; compile and CUDA Graph disabled. |
| `compiled` | Same operators, vLLM piecewise `torch.compile` and CUDA Graph enabled. |
| `fast` | Compiled execution plus Triton fast Hadamard, fused permutation/activation fake quantization, and LCTC output affine. |

`compiled` and `fast` are experimental and must pass the GPU checks below in the
target environment. They retain the mathematical transforms but can change
floating-point reduction/rounding and quantization decisions. Dynamic/LongRoPE
snapshots require `cached`. Original HF models retain their previous eager
settings and ignore this snapshot-specific variable. All modes remain single-GPU
with prefix caching disabled. The resolved mode is recorded in evaluation metadata.

```bash
# Existing commands automatically use cached execution for saved snapshots.
conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4

# Experimental fast path; validate before using its benchmark scores.
conda activate loopquant-vllm && \
  LOOPQUANT_VLLM_EXECUTION=fast USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4
```

Both launchers use `python -s` to keep packages in `~/.local` out of the
runtime import path. Direct CLI examples use the same option. vLLM's registry
starts a separate Python interpreter without copying `-s`, so backend initialization
also temporarily sets `PYTHONNOUSERSITE=1` for its child processes. It also sets
`VLLM_WORKER_MULTIPROC_METHOD=spawn`: CUDA availability checks in the parent can
make `fork` unsafe even before PyTorch reports CUDA as initialized. EngineCore
and workers therefore use fresh interpreters instead of inheriting CUDA state.
Both environment variables are restored in the calling process when initialization
finishes or fails; no shell export is needed. Python API callers must create the
backend inside an `if __name__ == "__main__":` guard; the CLI already does this.
Engine seed, context budget, task batch sizes and truncation policy follow the
evaluation protocol. vLLM always uses its KV cache, so `generative.use_cache: false`
is rejected rather than silently ignored. `CUDA_VISIBLE_DEVICES` selects the GPU;
the logical device must be `cuda:0` (or `auto`). Engine arguments, skipped suites,
versions and per-request generation receipts are saved. Preflight and model-load
failures also produce a failed manifest.

CPU reference tests cover tensor restoration, RoPE, recurrent prefill and
incremental decode with W4A4/mixed A4/A8 and non-neutral LCTC/Norm parameters,
shared matrix caches, and a context-free tensor graph. They do **not** exercise the real
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

# Repeat snapshot checks in each mode; includes standalone Triton operator checks.
LOOPQUANT_VLLM_EXECUTION=fast LOOPQUANT_VLLM_TEST_MODEL=/path/to/loopquant \
  LOOPQUANT_VLLM_COMPARE_HF=1 python -s -m pytest tests/test_vllm_gpu.py -q

# Warmup is excluded. Record engine TTFT/decode timing when available, plus
# same-process reference/cached/fast operator times and extra peak allocations.
# Run in separate processes for cached, compiled and fast with identical settings.
LOOPQUANT_VLLM_EXECUTION=cached LOOPQUANT_VLLM_TEST_MODEL=/path/to/loopquant \
  LOOPQUANT_VLLM_BENCHMARK=1 \
  LOOPQUANT_VLLM_BENCHMARK_OUTPUT=outputs/vllm_cached_benchmark.json \
  python -s -m pytest tests/test_vllm_gpu.py -q -s
```

## Checkpoint format

`--save` writes `output-dir/checkpoint` **before any evaluation task/scorer is
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

## FP activation analysis

This separate entry point observes original FP model activations only. It does not
quantize, fit low-rank factors, run lm-eval, or modify quantization configurations.
It compares the same token positions at the output of each physical decoder block,
before the outer loop's final normalization. The model follows all requested loops
with early exit and KV caching disabled.

```bash
# Optional plotting dependency; --no-plots exports JSON/CSV without it.
conda activate loopquant && \
  python -m pip install --no-user -i https://mirrors.aliyun.com/pypi/simple/ matplotlib

# Collect from an original model, then analyze (BF16 by default).
# --model can also be an original local HF model directory.
conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_activations.sh ouro_1.4b_bf16 \
    --model ByteDance/Ouro-1.4B --num-loops 4

# Reuse an existing FP teacher cache without loading a model or dataset.
# Select the actual teacher-key directory containing manifest.json.
conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/analyze_activations.sh ouro_1.4b_bf16 \
    --teacher-cache 'cache/calibration/ouro_1.4b/<teacher_key>'
```

`LOOPQUANT_PYTHON` can select another interpreter; the launcher uses `python -s`.
Results go to `outputs/<experiment-name>/activation_analysis/`:

```text
analysis.json                  # provenance, selected positions and all metrics
cosine_similarity.csv          # every layer and every pair of loops
adjacent_changes.csv           # relative change and r90 for adjacent loops
channel_statistics.csv         # per-channel RMS, energy share, previous-loop cosine
channel_overlap.csv            # per-layer adjacent-loop Top-K overlap and K
figures/adjacent_changes.png    # layer-by-loop-transition heatmaps
figures/adjacent_changes.svg
figures/channel_rms.png         # standalone wide RMS heatmaps for selected layers
figures/channel_rms.svg
figures/channel_changes.png     # channel cosine and Top-K overlap for selected layers
figures/channel_changes.svg
run.log
launcher.log
```

The `adjacent_changes` figure shows adjacent-loop cosine, relative change and increment r90.
Full pairwise cosine values remain in JSON/CSV. Regenerating plots also removes
obsolete `figures/cosine_similarity.png` and `.svg` from that report directory.

The three block-level metrics use output X_t with aligned token rows:

- **Cosine similarity:** average token-wise cosine for every pair of loops.
  Zero-norm token pairs are excluded and their valid counts are recorded. An
  entirely undefined pair is JSON `null` / an empty CSV cell.
- **Relative change:** ||X_t - X_(t-1)||_F / ||X_(t-1)||_F computed separately
  for each sample, then averaged across samples. Zero-to-zero is 0;
  zero-to-nonzero is undefined and excluded with the valid sample count recorded.
- **Increment r90:** the minimum rank explaining at least 90% of the squared
  singular-value energy of X_t - X_(t-1), pooling aligned token rows across the
  selected samples. It uses the eigenvalues of the smaller Gram matrix in FP32,
  without random SVD or mean centering. An exactly zero increment has rank 0.

Channel statistics reuse the same loaded FP outputs, without another forward or
another pass over the teacher cache. Every layer and channel is saved in the
per-layer `channels` object in `analysis.json` and the two channel CSVs:

- **Channel RMS and energy share:** RMS over all selected sample/token rows for
  each loop/channel. Energy share is squared RMS divided by the sum of squared
  RMS across channels; an all-zero loop has undefined shares.
- **Channel response cosine:** adjacent-loop cosine for the same channel over
  aligned token rows, without centering. This fixes the channel and compares its
  token response, whereas the block metric fixes a token and compares its channel
  vector. Zero-norm channel responses are undefined. The first loop's CSV entry
  is empty because it has no previous loop.
- **High-energy channel overlap:** intersection size / K for adjacent Top-K sets,
  with `K = max(1, ceil(channel_topk_ratio * width))`. Default
  `--channel-topk-ratio 0.01` selects the highest 1%. Ties use ascending channel IDs
  for deterministic selection. If either loop is all-zero the overlap is undefined;
  tied/flat energy profiles should be interpreted together with their RMS values.

The standalone `channel_rms` figure has one heatmap per selected layer, stacking
layers vertically and using the full figure width for channels. PNG width grows
with channel count, leaving about two horizontal pixels per channel at 180 DPI.
Nearest-neighbor rendering without resampling preserves narrow bright peaks;
view the PNG at native resolution or zoom the SVG to inspect individual channels.
RMS retains its linear scale, with a common scale across loops within each layer.

The `channel_changes` figure has one row per selected physical layer and two
columns: the response-cosine heatmap (channel by loop transition) and the Top-K
overlap curve. Channel IDs keep their original order in every loop; no per-loop
sorting is applied to either figure's heatmaps. Channel cosine uses a common
scale across displayed layers. Each plot has its own labeled scale.

By default only the first, middle (`num_layers // 2 + 1`) and last layers are
plotted, with duplicates removed for small models. Use `--plot-layers 1 13 24`,
for example, to choose 1-based layer numbers. This changes only the two channel
figures; all layers are still included in statistics and the original block figure.
Neither optional argument requires editing the shell launcher. Undefined channel
values are JSON `null` / empty CSV cells. Channel IDs are 0-based.

Defaults are `--num-samples 16 --sequence-length 2048 --batch-size 4
--tokens-per-sample 128 --seed 0 --device auto`. Original models use WikiText-2
training windows and accept `--dtype bfloat16|float16|float32`; all statistics use
FP32. Only sampled outputs are temporarily spooled to disk, preserving the model
dtype, and removed after analysis. Existing FP caches are strictly read-only;
their loop count and actual sequence length/dtype are used. The cache mode selects
up to the requested number of samples reproducibly, and does not load its saved
weights or create Hessian/clip caches. `--num-loops`, if supplied, must match.

Token positions are sampled uniformly without replacement per sequence, sorted,
and reused in every layer/loop. The token budget is capped by sequence length;
sample IDs and positions are saved in `analysis.json`. Keep these budgets and seeds
fixed when comparing results: r90 depends on the sampled matrix dimensions, and
cache collection batch sizes can also affect floating-point rounding. Layers and
loops in result tables are 1-based; token positions and sample IDs are 0-based.
Falling update magnitudes or low increment ranks are observations, not by
themselves evidence of better model quality or better low-rank compensation.

## Low-rank compensation

Select `algorithm: low_rank` using the new `configs/ouro_*_low_rank.yaml` files.
Existing YAMLs default to the original CLDC -> ILPA -> LCTC algorithm. The shell
command is unchanged:

```bash
conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/quantize.sh configs/ouro_1.4b_w4a4_low_rank.yaml

conda activate loopquant && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/quantize.sh configs/ouro_1.4b_w4a8_low_rank.yaml

conda activate loopquant && \
  USE_VLLM=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4_low_rank

conda activate loopquant-vllm && \
  USE_VLLM=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh ouro_1.4b_w4a4_low_rank
```

The launcher reads the saved `num_loops` automatically; direct CLI calls must
supply a matching `--num-loops`. Low-rank vLLM currently uses `cached` execution; the
original LoopQuant `compiled`/`fast` modes are not enabled for this new method.
No dependency changes or low-bit kernels are required for this path.

Each of the seven attention/MLP projections contains one shared W4/G32 base Q
and an independent rank-k pair B_t A_t for each loop. `method.rank_step` is k;
`num_loops` is T, with Tk(m+n) low-rank parameters for a matrix of shape (m,n).
Embeddings, LM head, norms and biases retain their original floating precision.

The fitter minimizes the weighted sum of linear-output reconstruction errors
`||(Q + B_t A_t) Z_t - Y_t||² / N_t`. X_t and Y_t come from the FP teacher;
Z_t is the fake-quantized/dequantized X_t using a fixed, offline-searched clip.
Bias is subtracted from Y_t when present. This is a per-Linear objective, not
decoder-block reconstruction or end-to-end training.

1. Cache immutable FP teacher trajectories and original projection weights.
2. Fit per-module/per-loop clipping by activation MSE. Activation quantization
   remains dynamic symmetric G32: [-7,7] for A4 or [-127,127] for A8. Both the
   base and low-rank branch consume the same Z_t; the correction is not requantized.
3. Replay each unmodified FP block with the original teacher collection batch
   shapes. With original weight W0, form R0_t=Y_t-W0 Z_t directly and collect
   H_t=Z_t Z_t'/N_t, K_t=R0_t Z_t'/N_t and ||R0_t||²/N_t entirely in FP64.
   Activation rounding still uses the original activation dtype. Centering avoids
   subtracting large Y-energy terms when scoring a nearly reconstructed layer.
4. Start with all low-rank factors zero. With the factors fixed, solve the shared
   continuous regression, then quantize with GPTQ using the Gram of Z. By default,
   ridge penalizes the Q update relative to FP W0 in the first round and the
   previous accepted Q in later rounds: W_new=W_ref+(D-W_ref H)(H+rho I)^-1,
   where D includes the current low-rank subtraction. Residual moments compute
   this without forming large D. With Q fixed, whiten each loop's residual cross
   moment and use exact truncated SVD to solve the regularized rank-k problem.
   By default, the low-rank ridge penalizes the whole product rho||L_t||².
   The two damping targets can be selected independently below.
5. Repeat complete Q/low-rank rounds, retaining the full Q/A/B combination with
   the lowest original, unregularized reconstruction error. Also score a baseline
   made by GPTQ of W0 plus ordinary weight-error SVD, using the same Z Gram and
   clips as the alternating candidates. This is not the former FP-X-Gram baseline.
   Stop if a round does not improve the preceding alternating round. A stronger
   baseline does not prevent trying later alternating rounds. best_iteration=0
   means the baseline won. Factors and the base are evaluated in their saved dtype;
   this linear statistical objective does not simulate intermediate BF16 GEMM
   rounding or account for upstream student-state drift.

Block Adam training, STE and temporary student-state propagation have been
removed. Ordinary weight-error SVD is only a comparison/fallback candidate;
alternating rounds start from zero low-rank factors. The configuration is:

```yaml
method:
  sharing: independent
  rank_step: 16
  solver:
    weight_update: regression_gptq
    iterations: 3
    damp_percent: 0.001
    quantized_damping_target: update
    low_rank_damping_target: weight
    loop_weights: null
    block_size: 128
    statistics_tokens_per_loop: null
```

`iterations: 1` performs one Q update followed by one weighted low-rank fit.
Three rounds are a starting setting, not a validated Ouro optimum. `loop_weights`
is null for equal weights or a positive list of length T; weights are normalized.
`block_size` is the GPTQ processing width, not quantization group size.

`quantized_damping_target` and `low_rank_damping_target` independently choose `weight`
or `update`. For Q, these add rho||Q||² or rho||Q-Q_ref||² to the continuous
regression objective; Q_ref is FP W0 initially and the previous accepted Q later.
For each low-rank branch, they add rho||L_t||² or rho||L_t-L_t,prev||². The latter
uses zero initially, so both low-rank modes coincide in the first round. Its
closed-form fit uses H+rho I and C+rho L_t,prev; the entire new L_t remains rank k,
not just its increment. This regularizes the product B_t A_t, not each factor.
Both targets use the same `damp_percent`, with their respective Gram scale.
Missing fields retain the defaults above, including when loading older configs.
Targets are saved in checkpoint config and run reports; changing them reuses
teacher, clipping and linear-statistics caches. Candidate selection still uses
the original reconstruction error without ridge, including the fallback baseline.

The relative ridge is `rho = damp_percent * mean(diag(H))`, computed separately
for the shared Gram and each loop Gram (diagonal means have a 1e-8 numerical
floor for zero inputs). The default relative damping is 0.001 for both updates.
Gram accumulation, regression, Cholesky, SVD and objective scoring use FP64;
GPTQ column feedback remains FP32 and reuses the inverse factor without adding
damping twice. Cholesky checks finite inputs and retries with increasing ridge
only up to max(requested_percent,0.01) times the diagonal mean, with at most eight
attempts. Any increase is logged with projection/loop context; run reports store
the actual shared and per-loop ridge values. Persistent failures report matrix
shape, diagonal range and attempted damping. The reported loss excludes ridge,
so a worsening complete round is rejected. Exact SVD and FP64 statistics can
increase calibration time and memory; this does not change inference precision.

`sharing: prefix` and `solver.weight_update: gptaq` are reserved; selecting them
for quantization fails before collecting a teacher cache. Old prefix checkpoints
still load for HF and vLLM inference. Old quantization YAMLs must be migrated to
the new fields; old checkpoint optimizer metadata is migrated only during loading.
The former `hessian_mode: mean_activation` ablation is not part of this objective.

All calibration tokens are eligible; `statistics_tokens_per_loop: null` uses all
tokens for clip and linear statistics. A cap takes the same deterministic token
prefix in each loop and changes the statistics key. Activation precision is uniform
across all projections and loops: `w4a4` uses A4 and `w4a8` uses A8. Low-rank
configurations do not accept per-loop bit plans and never run ILPA.
`rotation.enabled` and `channel_scaling.enabled` are reserved and must remain false.

`calibration.batch_size` controls FP teacher collection; its default is 4.
Collection batches equal-length
sequences, retains the original per-sample cache layout, and overlaps bounded
background disk writes/checksums with forward execution. GPU-to-CPU snapshots
remain synchronous. On Ouro, only the final token's unused vocabulary logits are
computed; all decoder token states are still collected. Progress logs distinguish
weight fingerprinting, original-weight writes, forward batches and completed writes.
Changing collection batch size can change floating-point rounding and therefore
uses a separate teacher key. Set it to 1 to reuse an existing single-sample cache.

The default `calibration.cache_dir` is `cache/calibration`, relative to the working
directory (the launchers enter the repository). The layout is:

```text
cache/calibration/
  tokens/                                      # tokenizer-keyed calibration IDs
  <model_key>/<teacher_key>/
    manifest.json
    tokens.safetensors
    teacher/block_000/weights.safetensors
    teacher/block_000/loop_00/shard_00000.safetensors
    activation_clip/<clip_key>/block_000.safetensors
    linear_statistics/<statistics_key>/block_000.safetensors
```

Teacher keys include original weights, actual token IDs, tokenizer, source code,
model configuration, loop count and precision. They do not include ranks,
sharing or solver settings. Derived statistics have their own keys and
checksums, including activation bits, clipping and token selection. They can be
reused when ranks, loop weights, damping or alternating rounds change. The FP64
residual-moment cache uses version 3; old FP32 linear statistics are recomputed,
while immutable FP teacher trajectories and the version-2 clip cache are reused.
Teacher inputs/outputs retain their dtype and are streamed to disk;
full trajectories and per-loop Grams can require substantial disk space.
Neither calibration caches nor the original weights are
needed to load a finished checkpoint.

Snapshots go to `output_dir/checkpoint` before evaluation. Their manifest
identifies `low-rank-fake-quantized` format version 1, factor layout and loop plan;
HF and vLLM dispatch to the correct restorer automatically. The saved weights
are floating fake-quantized tensors, not packed INT4. Use distinct config filenames
or `quantize.sh --tag` values for solver/rank ablations. Existing LoopQuant
snapshots remain valid and can be loaded by the direct CLI at their original paths.

## License

This work is released under the Apache 2.0 license.
