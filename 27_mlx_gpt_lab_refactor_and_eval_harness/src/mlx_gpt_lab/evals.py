from __future__ import annotations

import json
from pathlib import Path

from .generate import generate_text, load_model_tokenizer_from_run
from .utils import append_jsonl, load_json, resolve_path, write_json


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            prompts.append(json.loads(line))
    return prompts


def repeated_char_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(text, text[1:]) if a == b)
    return repeats / max(len(text) - 1, 1)


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    if len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def run_evaluation(project_dir: Path, run_dir: Path, eval_config_path: Path) -> Path:
    eval_config = load_json(eval_config_path)
    model, tokenizer, train_config, checkpoint_path = load_model_tokenizer_from_run(project_dir, run_dir)
    prompts_path = resolve_path(project_dir, eval_config.get("prompts_path", "evals/prompts_zh.jsonl"))
    prompts = load_prompts(prompts_path)

    output_dir = run_dir / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.jsonl"
    if results_path.exists():
        results_path.unlink()

    lengths = []
    empty_count = 0
    repeated_chars = []
    repeated_ngrams = []
    bad_output_count = 0

    generation_config = {**train_config, **eval_config}
    for item in prompts:
        generated = generate_text(
            model,
            tokenizer,
            item["prompt"],
            generation_config,
            max_new_tokens=int(eval_config.get("max_new_tokens", 60)),
        )
        continuation = generated[len(item["prompt"]) :]
        row = {
            **item,
            "generated": generated,
            "continuation": continuation,
            "length": len(continuation),
            "repeated_char_ratio": repeated_char_ratio(continuation),
            "repeated_3gram_ratio": repeated_ngram_ratio(continuation, 3),
            "has_bad_marker": "�" in continuation or "<unk>" in continuation,
        }
        append_jsonl(results_path, row)
        lengths.append(row["length"])
        empty_count += int(row["length"] == 0)
        repeated_chars.append(row["repeated_char_ratio"])
        repeated_ngrams.append(row["repeated_3gram_ratio"])
        bad_output_count += int(row["has_bad_marker"])

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "prompts_path": str(prompts_path),
        "num_prompts": len(prompts),
        "avg_generation_length": sum(lengths) / max(len(lengths), 1),
        "empty_output_count": empty_count,
        "avg_repeated_char_ratio": sum(repeated_chars) / max(len(repeated_chars), 1),
        "avg_repeated_3gram_ratio": sum(repeated_ngrams) / max(len(repeated_ngrams), 1),
        "bad_output_count": bad_output_count,
    }
    write_json(output_dir / "eval_metrics.json", metrics)
    return output_dir
