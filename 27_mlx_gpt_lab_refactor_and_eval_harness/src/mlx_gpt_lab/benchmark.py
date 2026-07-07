from __future__ import annotations

import time
from pathlib import Path

from .generate import generate_ids, load_model_tokenizer_from_run
from .utils import count_parameters, write_json


def run_benchmark(project_dir: Path, run_dir: Path, prompt: str = "人工智能", max_new_tokens: int = 80) -> Path:
    model, tokenizer, config, checkpoint_path = load_model_tokenizer_from_run(project_dir, run_dir)
    prompt_ids = tokenizer.encode(prompt)

    start = time.perf_counter()
    output_ids = generate_ids(model, prompt_ids, config, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - start
    new_tokens = max(0, len(output_ids) - len(prompt_ids))

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": new_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_second": new_tokens / max(elapsed, 1e-9),
        "parameter_count": count_parameters(model.parameters()),
        "note": "本 benchmark 是教学用小模型单次生成测速，不代表稳定工业吞吐。",
    }
    output_dir = run_dir / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "benchmark.json", metrics)
    return output_dir
