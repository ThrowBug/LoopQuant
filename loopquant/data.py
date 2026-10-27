"""Deterministic calibration data construction."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile

import torch
from datasets import load_dataset


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    descriptor = {"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map,
                  "backend": backend.to_str() if backend is not None else tokenizer.init_kwargs,
                  "add_special_tokens": False}
    sentencepiece = getattr(tokenizer, "sp_model", None)
    if sentencepiece is not None:
        descriptor["sentencepiece"] = hashlib.sha256(sentencepiece.serialized_model_proto()).hexdigest()
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _token_hash(ids):
    return hashlib.sha256(memoryview(ids.contiguous().numpy())).hexdigest()


def build_calibration_sequences(
    tokenizer,
    texts: list[str],
    num_samples: int = 128,
    sequence_length: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """Tokenize documents independently, concatenate, and sample deterministic windows."""
    chunks = []
    for text in texts:
        if not str(text).strip():
            continue
        tokens = tokenizer(str(text), add_special_tokens=False, return_tensors="pt").input_ids[0]
        if tokens.numel():
            chunks.append(tokens)
    if not chunks:
        raise RuntimeError("calibration split produced no tokens")
    tokens = torch.cat(chunks)
    if tokens.numel() < sequence_length:
        raise RuntimeError(f"dataset has {tokens.numel()} tokens; need {sequence_length}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    upper = tokens.numel() - int(sequence_length)
    starts = torch.randint(0, upper + 1, (int(num_samples),), generator=generator)
    return torch.stack([tokens[int(start) : int(start) + sequence_length] for start in starts])


def load_calibration_data(
    tokenizer,
    dataset: str = "wikitext2",
    split: str = "train",
    num_samples: int = 128,
    sequence_length: int = 2048,
    seed: int = 0,
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """Load calibration data only; evaluation splits are never used for method selection."""
    normalized = dataset.lower().replace("-", "")
    if normalized != "wikitext2":
        raise ValueError("the validated calibration path supports WikiText-2 only")
    cache_file = None
    if cache_dir is not None:
        tokenizer_key = tokenizer_fingerprint(tokenizer)
        cache_file = Path(cache_dir) / f"wikitext2_{split}_{num_samples}_{sequence_length}_{seed}_{tokenizer_key}.pt"
        if cache_file.exists():
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            ids = payload["input_ids"]
            if tuple(ids.shape) != (num_samples, sequence_length) or ids.dtype != torch.long:
                raise ValueError(f"invalid token cache shape or dtype: {cache_file}")
            if payload.get("tokens_sha256") != _token_hash(ids):
                raise ValueError(f"calibration token cache checksum mismatch: {cache_file}")
            return ids
    corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    sequences = build_calibration_sequences(
        tokenizer, list(corpus["text"]), num_samples, sequence_length, seed
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tokens-", dir=cache_file.parent) as temporary:
            staged = Path(temporary) / cache_file.name
            torch.save(
                {
                    "input_ids": sequences,
                    "tokens_sha256": _token_hash(sequences),
                    "dataset": "wikitext/wikitext-2-raw-v1",
                    "split": split,
                    "seed": int(seed),
                    "sequence_length": int(sequence_length),
                },
                staged,
            )
            staged.replace(cache_file)
    return sequences
