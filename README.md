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

Both scripts explicitly enable all four evaluation suites. `quantize.sh` also
explicitly enables saving. Edit their flag lists to select a different combination.

```bash
# Quantize -> save -> evaluate. Reads model_path and output_dir from the run YAML.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 bash scripts/quantize.sh configs/ouro_1.4b_w4a4.yaml

# Original HF model ID or local HF model directory.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh ByteDance/Ouro-1.4B 4
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh /models/Ouro-1.4B 4

# Saved fake-quantized model, with no original weights or calibration needed.
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh outputs/ouro_1.4b_w4a4/loopquant 4
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
they do not create isolation. Neither shell script silently sets the environment
variable. The scorer self-check runs when evaluation tasks are loaded.

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
