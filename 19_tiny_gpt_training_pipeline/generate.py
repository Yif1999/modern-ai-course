from __future__ import annotations

import argparse

from config import OUTPUT_DIR, TrainConfig, ensure_project_dirs
from model import TinyGPT, generate_ids
from tokenizer import CharacterTokenizer
from utils import load_checkpoint, set_seed


def parse_top_k(value: str):
    if value.lower() in {"none", "null", "0"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["latest", "best"], default="best")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prompt", default="hello ")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--strategy", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=parse_top_k, default=8)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    config = TrainConfig()
    ensure_project_dirs()
    set_seed(config.seed)

    tokenizer = CharacterTokenizer.load(config.vocab_path)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )

    meta = load_checkpoint(
        model,
        optimizer=None,
        kind=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
    )

    ids = generate_ids(
        model,
        tokenizer.encode(args.prompt),
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    text = tokenizer.decode(ids)

    output_path = OUTPUT_DIR / "generated_from_generate_py.txt"
    output_path.write_text(text, encoding="utf-8")

    print("=== Tiny GPT Generate ===")
    print("Checkpoint:", meta["model_path"])
    print("Prompt:", repr(args.prompt))
    print("Output saved:", output_path)
    print()
    print(text)


if __name__ == "__main__":
    main()
