from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from model import TinyGPT, generate_ids
from tokenizer_bpe import BPETokenizer, TOKENIZER_PATH


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CONFIG_PATH = OUTPUT_DIR / "config.json"
GENERATED_PATH = OUTPUT_DIR / "generated_from_generate_bpe.txt"


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    num_heads: int
    num_layers: int


def load_config() -> ModelConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return ModelConfig(
        vocab_size=int(payload["vocab_size"]),
        block_size=int(payload["block_size"]),
        n_embd=int(payload["n_embd"]),
        num_heads=int(payload["num_heads"]),
        num_layers=int(payload["num_layers"]),
    )


def resolve_checkpoint(kind: str, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return checkpoint_path
    pointer = CHECKPOINT_DIR / f"{kind}.json"
    meta = json.loads(pointer.read_text(encoding="utf-8"))
    return meta["model_path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="人工智能")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    mx.random.seed(42)
    np.random.seed(42)

    config = load_config()
    tokenizer = BPETokenizer(Tokenizer.from_file(str(TOKENIZER_PATH)))
    model = TinyGPT(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_path)
    model.load_weights(checkpoint, strict=True)

    start_ids = tokenizer.encode(args.prompt, add_bos=True)
    ids = generate_ids(
        model,
        start_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(ids)
    GENERATED_PATH.write_text(text, encoding="utf-8")

    print("=== Chinese BPE Tiny GPT Generate ===")
    print("Checkpoint:", checkpoint)
    print("Tokenizer:", TOKENIZER_PATH)
    print("Prompt:", args.prompt)
    print("Output:", GENERATED_PATH)
    print()
    print(text)


if __name__ == "__main__":
    main()
