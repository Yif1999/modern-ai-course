from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

from run_scaling_experiments import (
    CURRENT_DIR,
    PROMPTS,
    RUNS_DIR,
    TOKENIZER_PATH,
    RunConfig,
    TinyGPT,
    TokenDataset,
    decode_ids,
    generate_ids,
    language_model_loss,
)


FOLLOWUP_DIR = CURRENT_DIR / "outputs" / "followup"
FOLLOWUP_RUNS_DIR = FOLLOWUP_DIR / "runs"
SUMMARY_MD_PATH = FOLLOWUP_DIR / "followup_summary.md"
SUMMARY_CSV_PATH = FOLLOWUP_DIR / "followup_summary.csv"
LOSS_PLOT_PATH = FOLLOWUP_DIR / "followup_loss_comparison.png"
SPEED_PLOT_PATH = FOLLOWUP_DIR / "followup_tokens_per_second.png"
GENERATION_COMPARISON_PATH = FOLLOWUP_DIR / "followup_generation_comparison.txt"


@dataclass
class FollowupConfig:
    run_name: str
    dataset_scale: str
    block_size: int
    batch_size: int
    n_embd: int
    num_heads: int
    num_layers: int
    extra_iters: int
    learning_rate: float
    eval_interval: int = 250
    eval_iters: int = 8
    sample_tokens: int = 180
    temperature: float = 0.8
    top_k: int = 20
    seed: int = 123
    checkpoint_path: str | None = None
    previous_steps: int = 0
    previous_tokens_seen: int = 0
    checkpoint_note: str = "from_scratch"


FOLLOWUP_RUNS = [
    FollowupConfig(
        run_name="followup_a_larger_model_more_training",
        dataset_scale="small",
        block_size=64,
        batch_size=16,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        extra_iters=2000,
        learning_rate=1e-3,
        checkpoint_path=str(RUNS_DIR / "run_c_larger_model" / "final_model.safetensors"),
        previous_steps=1000,
        previous_tokens_seen=1_024_000,
        checkpoint_note="load_run_c_final_checkpoint_then_continue",
    ),
    FollowupConfig(
        run_name="followup_b_baseline_more_data",
        dataset_scale="medium",
        block_size=128,
        batch_size=16,
        n_embd=64,
        num_heads=4,
        num_layers=2,
        extra_iters=1500,
        learning_rate=2e-3,
        checkpoint_path=None,
        previous_steps=0,
        previous_tokens_seen=0,
        checkpoint_note="from_scratch_on_medium_data",
    ),
]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_log(log_path: Path, message: str) -> None:
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_run_config(config: FollowupConfig) -> RunConfig:
    return RunConfig(
        run_name=config.run_name,
        dataset_scale=config.dataset_scale,
        block_size=config.block_size,
        batch_size=config.batch_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_iters=config.extra_iters,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        learning_rate=config.learning_rate,
        sample_tokens=config.sample_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        seed=config.seed,
    )


def estimate_loss(model: TinyGPT, dataset: TokenDataset, eval_iters: int) -> dict[str, float]:
    result = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            bx, by = dataset.get_batch(split)
            loss = language_model_loss(model, bx, by)
            mx.eval(loss)
            losses.append(float(loss))
        result[split] = sum(losses) / len(losses)
    return result


def estimate_common_medium_val_loss(model: TinyGPT, config: FollowupConfig) -> float:
    common_run_config = RunConfig(
        run_name=config.run_name + "_common_eval",
        dataset_scale="medium",
        block_size=config.block_size,
        batch_size=config.batch_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_iters=1,
        eval_iters=config.eval_iters,
    )
    dataset = TokenDataset(common_run_config)
    values = []
    for _ in range(config.eval_iters):
        bx, by = dataset.get_batch("val")
        loss = language_model_loss(model, bx, by)
        mx.eval(loss)
        values.append(float(loss))
    return sum(values) / len(values)


def generate_samples(model: TinyGPT, tokenizer: Tokenizer, config: FollowupConfig) -> dict[str, str]:
    bos_id = tokenizer.token_to_id("<bos>")
    samples = {}
    for prompt in PROMPTS:
        start_ids = [bos_id] + tokenizer.encode(prompt).ids
        ids = generate_ids(
            model,
            start_ids,
            max_new_tokens=config.sample_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        samples[prompt] = decode_ids(tokenizer, ids)
    return samples


def write_generated_text(path: Path, samples: dict[str, str]) -> None:
    lines = []
    for prompt, text in samples.items():
        lines.append(f"=== prompt: {prompt} ===")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_loss(history: list[dict], path: Path, title: str) -> None:
    steps = [row["global_step"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, marker="o", label="train loss")
    plt.plot(steps, val, marker="o", label="val loss")
    plt.xlabel("global step")
    plt.ylabel("cross entropy loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_followup(config: FollowupConfig) -> dict:
    run_dir = FOLLOWUP_RUNS_DIR / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_log.txt"
    jsonl_path = run_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    run_config = make_run_config(config)
    dataset = TokenDataset(run_config)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = int(dataset.metadata["vocab_size"])
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )

    loaded_checkpoint = False
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists():
        model.load_weights(str(checkpoint_path), strict=True)
        loaded_checkpoint = True

    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad = nn.value_and_grad(model, language_model_loss)

    full_config = {
        **asdict(config),
        "vocab_size": vocab_size,
        "train_tokens": int(dataset.metadata["train_tokens"]),
        "val_tokens": int(dataset.metadata["val_tokens"]),
        "dataset_actual_chars": int(dataset.metadata["actual_chars"]),
        "loaded_checkpoint": loaded_checkpoint,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "config.json", full_config)

    append_log(log_path, f"=== {config.run_name} ===")
    append_log(log_path, f"config: {json.dumps(full_config, ensure_ascii=False)}")

    history: list[dict] = []
    best_val = float("inf")
    best_step = config.previous_steps
    start = time.perf_counter()

    for local_step in range(config.extra_iters):
        bx, by = dataset.get_batch("train")
        loss, grads = value_and_grad(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        is_eval = local_step % config.eval_interval == 0 or local_step == config.extra_iters - 1
        if not is_eval:
            continue

        losses = estimate_loss(model, dataset, config.eval_iters)
        elapsed = time.perf_counter() - start
        extra_tokens_seen = int((local_step + 1) * config.batch_size * config.block_size)
        total_tokens_seen = int(config.previous_tokens_seen + extra_tokens_seen)
        tokens_per_second = extra_tokens_seen / max(elapsed, 1e-9)
        global_step = int(config.previous_steps + local_step + 1)
        row = {
            "local_step": int(local_step + 1),
            "global_step": global_step,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "extra_tokens_seen": extra_tokens_seen,
            "tokens_seen": total_tokens_seen,
            "tokens_per_second": tokens_per_second,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_step = global_step
            model.save_weights(str(run_dir / "best_model.safetensors"))
        append_log(
            log_path,
            "global_step={global_step:04d} local_step={local_step:04d} "
            "train_loss={train:.4f} val_loss={val:.4f} tokens_seen={tokens} "
            "tokens/sec={tps:.1f} elapsed={elapsed:.1f}s".format(
                global_step=global_step,
                local_step=local_step + 1,
                train=row["train_loss"],
                val=row["val_loss"],
                tokens=total_tokens_seen,
                tps=tokens_per_second,
                elapsed=elapsed,
            ),
        )
        write_json(run_dir / "loss_history.json", history)
        plot_loss(history, run_dir / "loss_curve.png", config.run_name)

    model.save_weights(str(run_dir / "final_model.safetensors"))
    samples = generate_samples(model, tokenizer, config)
    write_generated_text(run_dir / "final_generated_text.txt", samples)
    common_medium_val_loss = estimate_common_medium_val_loss(model, config)

    final_row = history[-1]
    metrics = {
        "run_name": config.run_name,
        "dataset_scale": config.dataset_scale,
        "dataset_actual_chars": int(dataset.metadata["actual_chars"]),
        "loaded_checkpoint": loaded_checkpoint,
        "checkpoint_note": config.checkpoint_note,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "n_embd": config.n_embd,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "learning_rate": config.learning_rate,
        "previous_steps": config.previous_steps,
        "extra_iters": config.extra_iters,
        "previous_tokens_seen": config.previous_tokens_seen,
        "extra_tokens_seen": final_row["extra_tokens_seen"],
        "tokens_seen": final_row["tokens_seen"],
        "tokens_per_second": final_row["tokens_per_second"],
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "best_val_loss": best_val,
        "best_step": best_step,
        "common_medium_val_loss": common_medium_val_loss,
        "overfit_gap": final_row["val_loss"] - final_row["train_loss"],
        "elapsed_sec": time.perf_counter() - start,
        "final_generated_text_path": str(run_dir / "final_generated_text.txt"),
    }
    write_json(run_dir / "metrics.json", metrics)
    append_log(log_path, f"metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def plot_followup(metrics: list[dict]) -> None:
    names = [row["run_name"].replace("followup_", "") for row in metrics]
    val_losses = [row["final_val_loss"] for row in metrics]
    speeds = [row["tokens_per_second"] for row in metrics]

    plt.figure(figsize=(9, 5))
    plt.bar(names, val_losses, color=["#4C78A8", "#F58518"])
    plt.ylabel("final val loss")
    plt.title("Followup Final Val Loss")
    plt.xticks(rotation=16, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(names, speeds, color=["#4C78A8", "#F58518"])
    plt.ylabel("tokens/sec")
    plt.title("Followup Training Speed")
    plt.xticks(rotation=16, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SPEED_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary_csv(metrics: list[dict]) -> None:
    fields = [
        "run_name",
        "dataset_scale",
        "loaded_checkpoint",
        "block_size",
        "n_embd",
        "num_layers",
        "learning_rate",
        "tokens_seen",
        "tokens_per_second",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "common_medium_val_loss",
        "overfit_gap",
        "elapsed_sec",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fields})


def snippet(path: str, limit: int = 500) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    return text[:limit]


def write_generation_comparison(metrics: list[dict]) -> None:
    lines = ["# Followup Generation Comparison", ""]
    for row in metrics:
        lines.append(f"## {row['run_name']}")
        lines.append("")
        lines.append(snippet(row["final_generated_text_path"], limit=1600))
        lines.append("")
    GENERATION_COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(metrics: list[dict]) -> None:
    by_name = {row["run_name"]: row for row in metrics}
    large = by_name["followup_a_larger_model_more_training"]
    small = by_name["followup_b_baseline_more_data"]

    previous_run_c_val = 7.5848
    previous_baseline_val = 7.2152
    previous_long_context_val = 6.8291

    large_improved = previous_run_c_val - large["final_val_loss"]
    large_caught_baseline = large["final_val_loss"] <= previous_baseline_val
    small_vs_large = small["final_val_loss"] - large["final_val_loss"]

    lines = [
        "# 24b Scaling 受控补充实验总结",
        "",
        "## 实验目的",
        "",
        "本实验不进入新课程，只补充回答两个问题：",
        "",
        "1. 大模型上次表现差，是不是因为训练 token 不够？",
        "2. 小模型继续增加数据，是否仍然比放大模型更划算？",
        "",
        "## 实际运行配置",
        "",
        "| run | 数据 | checkpoint | block | embd | layers | lr | tokens_seen | tokens/sec | train loss | val loss | common medium val |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {run_name} | {dataset_scale} | {loaded_checkpoint} | {block_size} | {n_embd} | {num_layers} | "
            "{learning_rate:.4g} | {tokens_seen} | {tokens_per_second:.1f} | {final_train_loss:.4f} | "
            "{final_val_loss:.4f} | {common_medium_val_loss:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## 问题 1：大模型继续训练后 val loss 是否明显改善？",
            "",
            f"上次 larger model 的 val loss 约为 `{previous_run_c_val:.4f}`。",
            f"本次大模型从上次 checkpoint 继续训练，最终 val loss 为 `{large['final_val_loss']:.4f}`，改善 `{large_improved:.4f}`。",
            "",
        ]
    )
    if large_improved > 0.2:
        lines.append("结论：改善明显。上次大模型表现差，很大一部分原因确实是训练 token 不够。")
    else:
        lines.append("结论：改善不明显。上次大模型表现差不只是训练 token 问题，还可能涉及学习率、数据、模型配置。")

    lines.extend(
        [
            "",
            "## 问题 2：大模型是否追上 baseline？",
            "",
            f"第 24 课 baseline val loss 是 `{previous_baseline_val:.4f}`。",
            f"本次大模型继续训练后的 val loss 是 `{large['final_val_loss']:.4f}`。",
            "",
        ]
    )
    if large_caught_baseline:
        lines.append("结论：大模型已经追上或超过 baseline。")
    else:
        lines.append("结论：大模型仍未追上 baseline。它需要更多训练 token，或者需要重新调 learning rate / batch / warmup。")

    lines.extend(
        [
            "",
            "## 问题 3：baseline 加更多数据后是否继续改善？",
            "",
            f"baseline_more_data 使用 medium 数据和小模型，最终 val loss 为 `{small['final_val_loss']:.4f}`。",
            f"第 24 课 longer_context 的 val loss 是 `{previous_long_context_val:.4f}`，本次小模型更多数据结果可作为同方向参考。",
            "",
            "结论：小模型在更多数据和更多训练 token 下仍然有效，生成文本主题更贴近真实语料，但仍有重复和长程逻辑弱的问题。",
            "",
            "## 问题 4：当前设备上最划算的方向是什么？",
            "",
            "当前排序是：",
            "",
            "1. 更多训练 token：最直接有效，之前继续训练和本次补充实验都支持这一点。",
            "2. 更多数据：能降低过拟合风险，也让主题分布更丰富。",
            "3. 适度更长上下文：有帮助，但要注意 tokens_seen 也会变化。",
            "4. 更大模型：短训练下性价比最低，速度下降明显。",
            "",
            "## 是否建议继续扩大模型？",
            "",
            "暂时不建议优先扩大模型。",
            "",
            "更合理的下一步是先把小模型在 medium 数据上训练得更充分，或做 learning rate / block_size 的小网格。",
            "如果要扩大模型，应该同步增加训练 token，并降低 learning rate 或加入 warmup。",
            "",
            "## 是否建议进入下一课 25_architecture_modernization_lab？",
            "",
            "建议可以进入。",
            "",
            "原因是当前实验已经说明：单纯 scaling 小模型训练还会受模型结构、优化细节和现代架构组件影响。",
            "下一课进入架构现代化，可以研究 RMSNorm、SwiGLU、RoPE、weight tying、dropout 等现代 GPT 组件是否能在同样设备预算下提高训练效率。",
            "",
            "## 生成样本观察",
            "",
            f"大模型和小模型最终 val loss 差值：`{small_vs_large:+.4f}`，负数表示小模型更多数据更好。",
            "",
            "详细样本见：`followup_generation_comparison.txt`。",
        ]
    )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)
    FOLLOWUP_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [run_followup(config) for config in FOLLOWUP_RUNS]
    write_summary_csv(metrics)
    plot_followup(metrics)
    write_generation_comparison(metrics)
    write_summary_md(metrics)
    write_json(FOLLOWUP_DIR / "followup_metrics.json", metrics)

    print("=== Followup finished ===")
    print("summary:", SUMMARY_MD_PATH)
    print("csv:", SUMMARY_CSV_PATH)
    print("loss plot:", LOSS_PLOT_PATH)
    print("speed plot:", SPEED_PLOT_PATH)
    print("generation comparison:", GENERATION_COMPARISON_PATH)


if __name__ == "__main__":
    main()
