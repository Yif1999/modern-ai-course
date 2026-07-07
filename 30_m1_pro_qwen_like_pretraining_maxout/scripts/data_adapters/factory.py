from __future__ import annotations

from .base import AdapterConfig
from .hf_dataset import HFDatasetAdapter
from .local_jsonl import LocalJsonlAdapter
from .local_text import LocalTextAdapter


def build_adapter(config: AdapterConfig):
    kind = str(config.options.get("adapter", "")).strip().lower()
    if kind == "local_jsonl" or (config.path and config.path.endswith(".jsonl")):
        return LocalJsonlAdapter(config)
    if kind == "local_text" or (config.path and config.path.endswith(".txt")):
        return LocalTextAdapter(config)
    if kind == "hf_dataset" or config.dataset_name:
        return HFDatasetAdapter(config)
    raise ValueError(f"Cannot infer adapter for source: {config.source_name}")
