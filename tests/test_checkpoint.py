import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from loopquant.checkpoint import save_quantized_checkpoint, load_quantized_checkpoint
from loopquant.lctc import LoopConditionedCGB, apply_lctc_calibration
from loopquant.models import configure_looped_model, load_model_for_evaluation
from loopquant.quantization import (
    QUANTIZATION_CONTEXT, rotate_ouro_backbone, wrap_quantized_modules, fake_quantize_weight,
)
from modeling_tiny_ouro import TinyOuroConfig, TinyOuroForCausalLM
from test_evaluation import tiny_model_and_tokenizer


def prepared_model(allocation):
    torch.manual_seed(42)
    model = TinyOuroForCausalLM(TinyOuroConfig()).to(torch.bfloat16)
    _, tokenizer = tiny_model_and_tokenizer()
    configure_looped_model(model, 4)
    model.model.inv_freq.add_(2)
    model.model.original_inv_freq.add_(3)
    model.model.max_seq_len_cached = 32
    QUANTIZATION_CONTEXT.configure(4, allocation)
    rotate_ouro_backbone(model)
    modules = wrap_quantized_modules(model)
    for module in modules.values():
        module.set_cldc_scale(torch.linspace(0.8, 1.2, module.in_features).to(module.weight.dtype))
        module.cldc_hadamard = True
        module.cldc_permutation = torch.arange(module.in_features - 1, -1, -1)
        module.weight.data = fake_quantize_weight(module.weight, 4)
    cgb = LoopConditionedCGB.from_model(model, 4)
    with torch.no_grad():
        for spec, entry in cgb.iter_entries():
            entry.clip.fill_(0.9 - spec.loop_id * 0.03)
            entry.clip_initial.fill_(0.95)
            entry.gain.fill_(1.0 + spec.loop_id * 0.01)
            entry.beta.fill_(0.01)
            entry.sigma.fill_(0.5)
    apply_lctc_calibration(model, cgb)
    model._loopquant_allocation = tuple(allocation)
    model._loopquant_model_path = "/original/weights/not/available"
    QUANTIZATION_CONTEXT.enabled = True
    return model.eval(), tokenizer


@pytest.mark.parametrize("allocation", [(4, 4, 4, 4), (8, 4, 8, 4)])
def test_checkpoint_roundtrip_and_new_process(tmp_path, allocation):
    model, tokenizer = prepared_model(allocation)
    ids = torch.tensor([[3, 4, 3]])
    with torch.no_grad():
        logits = model(ids).logits
        tokens = model.generate(ids, max_new_tokens=3, do_sample=False, use_cache=False)
    snapshot = tmp_path / "checkpoint"
    save_quantized_checkpoint(model, tokenizer, snapshot, max_shard_bytes=20_000)
    assert len(list(snapshot.glob("*.safetensors"))) > 1
    # Loading must restore from tensors, not perform another rotation or PTQ.
    restored, _ = load_quantized_checkpoint(snapshot, device="cpu")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    assert restored.lctc_cgb.entries[next(iter(restored.lctc_cgb.entries))].clip.dtype == torch.float32
    assert torch.equal(model.model.inv_freq, restored.model.inv_freq)
    assert torch.equal(model.model.original_inv_freq, restored.model.original_inv_freq)
    assert restored.model.max_seq_len_cached == 32
    with torch.no_grad():
        torch.testing.assert_close(logits, restored(ids).logits, rtol=0, atol=0)
    assert QUANTIZATION_CONTEXT.activation_bits == allocation
    output = tmp_path / "restored.pt"
    script = (
        "import sys,torch; from loopquant.checkpoint import load_quantized_checkpoint; "
        "m,t=load_quantized_checkpoint(sys.argv[1],device='cpu'); ids=torch.tensor([[3,4,3]]); "
        "torch.save({'logits':m(ids).logits.detach(),'tokens':m.generate(ids,max_new_tokens=3,"
        "do_sample=False,use_cache=False)},sys.argv[2])"
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_MODULES_CACHE=str(tmp_path / "modules"))
    completed = subprocess.run([sys.executable, "-c", script, str(snapshot), str(output)],
                               env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = torch.load(output, weights_only=True)
    torch.testing.assert_close(logits, result["logits"], rtol=0, atol=0)
    assert torch.equal(tokens, result["tokens"])


def test_save_does_not_overwrite_and_checks_active_state(tmp_path):
    model, tokenizer = prepared_model((8, 4, 8, 4))
    QUANTIZATION_CONTEXT.set_allocation([4] * 4)
    with pytest.raises(RuntimeError, match="context"):
        save_quantized_checkpoint(model, tokenizer, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    QUANTIZATION_CONTEXT.set_allocation(model._loopquant_allocation)
    save_quantized_checkpoint(model, tokenizer, tmp_path / "checkpoint")
    with pytest.raises(FileExistsError):
        save_quantized_checkpoint(model, tokenizer, tmp_path / "checkpoint")


def test_corrupt_snapshot_never_falls_back_to_original(tmp_path):
    model, tokenizer = prepared_model((4, 4, 4, 4))
    path = save_quantized_checkpoint(model, tokenizer, tmp_path / "checkpoint")
    weights = next(path.glob("*.safetensors"))
    weights.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_model_for_evaluation(path, num_loops=4, device="cpu")


def test_parameter_aliases_and_snapshot_overrides(tmp_path):
    model, tokenizer = prepared_model((4, 4, 4, 4))
    model.lm_head.weight = model.model.embed_tokens.weight
    path = save_quantized_checkpoint(model, tokenizer, tmp_path / "checkpoint")
    loaded, _ = load_quantized_checkpoint(path, device="cpu")
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    with pytest.raises(ValueError, match="omit dtype overrides"):
        load_model_for_evaluation(path, num_loops=4, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="conflicts with saved"):
        load_model_for_evaluation(path, num_loops=2, device="cpu")
    evaluated, _, metadata = load_model_for_evaluation(path, num_loops=4, device="cpu")
    assert evaluated._loopquant_num_loops == 4
    assert metadata["mode"] == "quantized"


def test_failed_save_is_not_published(tmp_path, monkeypatch):
    import loopquant.checkpoint as checkpoint
    model, tokenizer = prepared_model((4, 4, 4, 4))
    monkeypatch.setattr(checkpoint, "_write_weights", lambda *a: (_ for _ in ()).throw(RuntimeError("disk error")))
    with pytest.raises(RuntimeError, match="disk error"):
        save_quantized_checkpoint(model, tokenizer, tmp_path / "checkpoint")
    assert list(tmp_path.iterdir()) == []
