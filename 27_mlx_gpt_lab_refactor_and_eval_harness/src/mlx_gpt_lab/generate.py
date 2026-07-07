from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .checkpoint import load_model_checkpoint
from .dataset import load_prepared_tokenizer
from .model_factory import create_model
from .sampling import sample_next_id
from .utils import load_json, resolve_path


def generate_ids(model, start_ids: list[int], config: dict, max_new_tokens: int | None = None) -> list[int]:
    ids = [int(i) for i in start_ids]
    if not ids:
        ids = [0]
    block_size = int(config["block_size"])
    max_new = int(max_new_tokens or config.get("max_new_tokens", 100))
    for _ in range(max_new):
        context = ids[-block_size:]
        idx = mx.array([context], dtype=mx.int32)
        logits = model(idx)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)
        next_id = sample_next_id(
            last_logits,
            temperature=float(config.get("temperature", 0.8)),
            top_k=int(config.get("top_k", 0)),
            top_p=float(config.get("top_p", 1.0)),
        )
        ids.append(next_id)
    return ids


def generate_text(model, tokenizer, prompt: str, config: dict, max_new_tokens: int | None = None) -> str:
    prompt_ids = tokenizer.encode(prompt)
    ids = generate_ids(model, prompt_ids, config, max_new_tokens=max_new_tokens)
    return tokenizer.decode(ids)


def load_model_tokenizer_from_run(project_dir: Path, run_dir: Path):
    config = load_json(run_dir / "config.json")
    tokenizer_type = config["tokenizer_type"]
    tokenizer_path = resolve_path(project_dir, config["tokenizer_path"])
    tokenizer = load_prepared_tokenizer(tokenizer_path, tokenizer_type)
    model = create_model(config, vocab_size=int(config["vocab_size"]))
    checkpoint_path = load_model_checkpoint(model, run_dir)
    mx.eval(model.parameters())
    return model, tokenizer, config, checkpoint_path
