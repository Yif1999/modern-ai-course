from __future__ import annotations

from .model_baseline import BaselineTinyGPT
from .model_qwen_dense import QwenDenseTinyGPT


def create_model(config: dict, vocab_size: int):
    model_type = config.get("model_type", "baseline_debug")
    common = {
        "vocab_size": vocab_size,
        "block_size": int(config["block_size"]),
        "n_embd": int(config["n_embd"]),
        "num_heads": int(config["num_heads"]),
        "num_layers": int(config["num_layers"]),
    }
    if model_type == "baseline_debug":
        return BaselineTinyGPT(**common)
    if model_type == "qwen_dense_tiny":
        return QwenDenseTinyGPT(
            **common,
            ffn_multiplier=float(config.get("ffn_multiplier", 3.0)),
            rope_base=float(config.get("rope_base", 10000.0)),
            weight_tying=bool(config.get("weight_tying", True)),
            use_bias=bool(config.get("use_bias", False)),
        )
    raise ValueError(f"未知 model_type: {model_type}")
