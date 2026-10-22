from types import SimpleNamespace

import torch

from loopquant.config import LoopQuantConfig
from loopquant.models import load_looped_transformer, load_model_for_evaluation
from loopquant.pipeline import LoopQuantPipeline
from loopquant.quantization import QUANTIZATION_CONTEXT
from modeling_tiny_ouro import TinyOuroConfig, TinyOuroForCausalLM
from test_evaluation import tiny_model_and_tokenizer


def test_hf_id_and_teacher_loader_preserve_student_allocation(monkeypatch):
    import loopquant.models as models
    sources = []
    monkeypatch.setattr(models.AutoConfig, "from_pretrained",
                        lambda source, **kw: sources.append(source) or TinyOuroConfig())
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(models.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **kw: TinyOuroForCausalLM(kw["config"]))
    QUANTIZATION_CONTEXT.configure(4, [8, 4, 8, 4])
    QUANTIZATION_CONTEXT.enabled = True
    teacher, _ = load_looped_transformer("ByteDance/Ouro-1.4B", device="cpu", num_loops=4, configure_context=False)
    assert sources == ["ByteDance/Ouro-1.4B"]
    assert QUANTIZATION_CONTEXT.activation_bits == (8, 4, 8, 4)
    assert QUANTIZATION_CONTEXT.enabled
    load_model_for_evaluation("ByteDance/Ouro-1.4B", device="cpu", num_loops=4)
    assert not QUANTIZATION_CONTEXT.enabled  # Full precision starts clean.


def test_pipeline_rebinds_and_retains_mixed_allocation(monkeypatch):
    import loopquant.pipeline as pipeline_module
    pipeline = LoopQuantPipeline(LoopQuantConfig.from_dict({"num_loops": 4}), device="cpu")
    student = SimpleNamespace(_loopquant_model_path="unused")
    allocation = [8, 4, 4, 8]
    monkeypatch.setattr(pipeline, "run_cldc", lambda *a: {})
    monkeypatch.setattr(pipeline, "run_ilpa", lambda *a: {"allocation": allocation})

    def teacher_loader(*args, **kwargs):
        assert kwargs["configure_context"] is False
        assert QUANTIZATION_CONTEXT.activation_bits == tuple(allocation)
        return object(), None

    def calibrate(*args):
        assert QUANTIZATION_CONTEXT.activation_bits == tuple(allocation)
        assert QUANTIZATION_CONTEXT.enabled
        return {}

    monkeypatch.setattr(pipeline_module, "load_looped_transformer", teacher_loader)
    monkeypatch.setattr(pipeline, "run_lctc", calibrate)
    QUANTIZATION_CONTEXT.configure(4)
    result = pipeline.run(student, None, torch.zeros(1, 2))
    assert result.model._loopquant_allocation == tuple(allocation)
    assert QUANTIZATION_CONTEXT.activation_bits == tuple(allocation)


def test_original_local_hf_directory_loads_without_quantization(tmp_path):
    torch.manual_seed(9)
    model = TinyOuroForCausalLM(TinyOuroConfig()).to(torch.bfloat16).eval()
    _, tokenizer = tiny_model_and_tokenizer()
    model.register_for_auto_class("AutoModelForCausalLM")
    model.config.register_for_auto_class("AutoConfig")
    original = tmp_path / "original"
    model.save_pretrained(original)
    tokenizer.save_pretrained(original)
    ids = torch.tensor([[3, 4]])
    with torch.no_grad():
        expected = model(ids).logits
    QUANTIZATION_CONTEXT.enabled = True
    restored, _, metadata = load_model_for_evaluation(original, device="cpu", num_loops=4)
    assert metadata["mode"] == "full_precision"
    assert not QUANTIZATION_CONTEXT.enabled
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(ids).logits, rtol=0, atol=0)
