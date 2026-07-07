from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .tokenizer import BPETokenizer, CharTokenizer, TokenizerLike, train_bpe_tokenizer
from .utils import resolve_path, write_json


FALLBACK_TEXT = """
人工智能正在改变我们的学习方式。
我们在本地 Mac 上使用 MLX 训练一个很小的中文 GPT。
模型看到前面的 token，然后预测下一个 token。
数据管线负责把原始文本变成 token ids。
tokenizer 把中文、English、数字123和标点符号转换成模型可以处理的编号。
训练时，我们观察 train loss、val loss、生成样本和 tokens/sec。
baseline_debug 用来做快速 smoke test。
qwen_dense_tiny 使用 RoPE、RMSNorm、SwiGLU 和 weight tying。
这个实验不是追求大模型效果，而是让每一步都可理解、可复现、可检查。
如果数据太少，模型会很快记住训练文本，但生成质量仍然有限。
后续我们会加入中文趣味语料、toy SFT 和更完整的评估工具。
""".strip()


@dataclass
class PreparedData:
    train_tokens_path: Path
    val_tokens_path: Path
    tokenizer_path: Path
    metadata_path: Path
    vocab_size: int
    train_tokens: int
    val_tokens: int


def ensure_raw_text(project_dir: Path, data_path: Path) -> str:
    if data_path.exists():
        return data_path.read_text(encoding="utf-8")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    text = ("\n" + FALLBACK_TEXT + "\n") * 80
    data_path.write_text(text, encoding="utf-8")
    return text


def prepare_data_from_config(config: dict, project_dir: Path, force: bool = False) -> PreparedData:
    tokenizer_type = config.get("tokenizer_type", "char")
    data_path = resolve_path(project_dir, config.get("data_path")) or (project_dir / "data/raw/fallback_zh.txt")
    processed_dir = resolve_path(project_dir, config.get("processed_dir")) or (project_dir / "data/processed/default")
    tokenizer_path = resolve_path(project_dir, config.get("tokenizer_path")) or (processed_dir / "tokenizer.json")

    processed_dir.mkdir(parents=True, exist_ok=True)
    text = ensure_raw_text(project_dir, data_path)

    train_tokens_path = processed_dir / "train_tokens.npy"
    val_tokens_path = processed_dir / "val_tokens.npy"
    metadata_path = processed_dir / "metadata.json"

    if not force and train_tokens_path.exists() and val_tokens_path.exists() and tokenizer_path.exists():
        tokenizer = load_prepared_tokenizer(tokenizer_path, tokenizer_type)
        train_tokens = np.load(train_tokens_path)
        val_tokens = np.load(val_tokens_path)
        return PreparedData(
            train_tokens_path=train_tokens_path,
            val_tokens_path=val_tokens_path,
            tokenizer_path=tokenizer_path,
            metadata_path=metadata_path,
            vocab_size=tokenizer.vocab_size,
            train_tokens=int(train_tokens.shape[0]),
            val_tokens=int(val_tokens.shape[0]),
        )

    if tokenizer_type == "char":
        tokenizer: TokenizerLike = CharTokenizer.train(text)
        tokenizer.save(tokenizer_path)
    elif tokenizer_type == "bpe":
        vocab_size = int(config.get("bpe_vocab_size", 1024))
        tokenizer = train_bpe_tokenizer(data_path, vocab_size, tokenizer_path)
    else:
        raise ValueError(f"未知 tokenizer_type: {tokenizer_type}")

    encoded = np.array(tokenizer.encode(text), dtype=np.int32)
    block_size = int(config["block_size"])
    if encoded.shape[0] < block_size * 20:
        encoded = np.tile(encoded, int(np.ceil((block_size * 20) / max(encoded.shape[0], 1))))

    split_idx = max(block_size + 2, int(encoded.shape[0] * 0.9))
    split_idx = min(split_idx, encoded.shape[0] - block_size - 2)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]
    if val_tokens.shape[0] <= block_size + 1:
        val_tokens = encoded[-(block_size * 4 + 2) :]

    np.save(train_tokens_path, train_tokens)
    np.save(val_tokens_path, val_tokens)

    metadata = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "data_path": str(data_path),
        "processed_dir": str(processed_dir),
        "tokenizer_path": str(tokenizer_path),
        "raw_characters": len(text),
        "total_tokens": int(encoded.shape[0]),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "block_size": block_size,
        "fallback_created": str(data_path).endswith("fallback_zh.txt"),
    }
    write_json(metadata_path, metadata)
    write_json(project_dir / "data/manifests" / f"{config['run_name']}_manifest.json", metadata)

    return PreparedData(
        train_tokens_path=train_tokens_path,
        val_tokens_path=val_tokens_path,
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
        vocab_size=tokenizer.vocab_size,
        train_tokens=int(train_tokens.shape[0]),
        val_tokens=int(val_tokens.shape[0]),
    )


def load_prepared_tokenizer(tokenizer_path: Path, tokenizer_type: str) -> TokenizerLike:
    if tokenizer_type == "char":
        return CharTokenizer.load(tokenizer_path)
    if tokenizer_type == "bpe":
        return BPETokenizer.load(tokenizer_path)
    raise ValueError(f"未知 tokenizer_type: {tokenizer_type}")


class BatchSampler:
    def __init__(
        self,
        train_tokens_path: Path,
        val_tokens_path: Path,
        block_size: int,
        batch_size: int,
        seed: int,
    ):
        self.train_tokens = np.load(train_tokens_path).astype(np.int32)
        self.val_tokens = np.load(val_tokens_path).astype(np.int32)
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def get_batch(self, split: str):
        data = self.train_tokens if split == "train" else self.val_tokens
        max_start = max(1, len(data) - self.block_size - 1)
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = np.stack([data[start : start + self.block_size] for start in starts])
        y = np.stack([data[start + 1 : start + self.block_size + 1] for start in starts])
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)
