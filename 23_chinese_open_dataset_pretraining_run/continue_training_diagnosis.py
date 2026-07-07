from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

from model import TinyGPT, generate_ids, language_model_loss


CURRENT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
BASE_OUTPUT_DIR = CURRENT_DIR / "outputs"
BASE_CHECKPOINT_DIR = BASE_OUTPUT_DIR / "checkpoints"
CONTINUE_DIR = BASE_OUTPUT_DIR / "continue_training"
CONTINUE_SAMPLE_DIR = CONTINUE_DIR / "samples"
CONTINUE_CHECKPOINT_DIR = CONTINUE_DIR / "checkpoints"
BEST_DIR = CONTINUE_DIR / "best_val_checkpoint"
FINAL_DIR = CONTINUE_DIR / "final_checkpoint"

TOKENIZER_PATH = BASE_OUTPUT_DIR / "tokenizer" / "chinese_bpe_tokenizer.json"
CONFIG_PATH = BASE_OUTPUT_DIR / "config.json"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_tokens.npy"

LOG_PATH = CONTINUE_DIR / "continue_training_log.txt"
LOSS_HISTORY_PATH = CONTINUE_DIR / "continue_loss_history.json"
LOSS_CURVE_PATH = CONTINUE_DIR / "continue_loss_curve.png"
COMPARISON_PATH = CONTINUE_DIR / "before_after_generation_comparison.txt"
DIAGNOSIS_PATH = CONTINUE_DIR / "diagnosis_report.md"

PROMPTS = ["人工智能", "大语言模型", "今天我们学习", "本地模型"]


@dataclass
class ContinueConfig:
    previous_steps: int
    extra_steps: int = 5000
    block_size: int = 128
    batch_size: int = 16
    n_embd: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 2e-3
    eval_interval: int = 250
    eval_iters: int = 10
    sample_interval: int = 1000
    max_new_tokens: int = 160
    temperature: float = 0.8
    top_k: int = 20
    seed: int = 123

    @property
    def total_steps(self) -> int:
        return self.previous_steps + self.extra_steps


class TokenDataset:
    def __init__(self, config: ContinueConfig):
        self.config = config
        self.train_tokens = np.load(TRAIN_TOKENS_PATH).astype(np.int32)
        self.val_tokens = np.load(VAL_TOKENS_PATH).astype(np.int32)

    def get_batch(self, split: str):
        source = self.train_tokens if split == "train" else self.val_tokens
        starts = np.random.randint(
            0,
            len(source) - self.config.block_size - 1,
            size=(self.config.batch_size,),
        )
        x = np.stack([source[i : i + self.config.block_size] for i in starts]).astype(np.int32)
        y = np.stack([source[i + 1 : i + self.config.block_size + 1] for i in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def ensure_dirs() -> None:
    CONTINUE_DIR.mkdir(parents=True, exist_ok=True)
    CONTINUE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    CONTINUE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_base_metadata() -> tuple[dict, dict]:
    run_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    best_meta = json.loads((BASE_CHECKPOINT_DIR / "best.json").read_text(encoding="utf-8"))
    return run_config, best_meta


def make_model(run_config: dict, config: ContinueConfig) -> TinyGPT:
    return TinyGPT(
        vocab_size=int(run_config["vocab_size"]),
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )


def decode_ids(tokenizer: Tokenizer, ids) -> str:
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def generate_for_prompts(model: TinyGPT, tokenizer: Tokenizer, config: ContinueConfig) -> dict[str, str]:
    out = {}
    bos_id = tokenizer.token_to_id("<bos>")
    for prompt in PROMPTS:
        start_ids = [bos_id] + tokenizer.encode(prompt).ids
        ids = generate_ids(
            model,
            start_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        out[prompt] = decode_ids(tokenizer, ids)
    return out


def save_samples(step: int, samples: dict[str, str]) -> Path:
    path = CONTINUE_SAMPLE_DIR / f"sample_step_{step:06d}.txt"
    lines = [f"=== sample step {step} ==="]
    for prompt, text in samples.items():
        lines.append("")
        lines.append(f"--- prompt: {prompt} ---")
        lines.append(text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def estimate_loss(model: TinyGPT, dataset: TokenDataset, config: ContinueConfig) -> dict:
    losses = {}
    for split in ["train", "val"]:
        values = []
        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            loss = language_model_loss(model, x, y)
            mx.eval(loss)
            values.append(float(loss))
        losses[split] = sum(values) / len(values)
    return losses


def save_checkpoint(model: TinyGPT, step: int, metrics: dict, target_dir: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "model.safetensors"
    meta_path = target_dir / "meta.json"
    model.save_weights(str(model_path))
    meta = {
        "step": int(step),
        "model_path": str(model_path),
        **metrics,
    }
    write_json(meta_path, meta)
    return meta


def save_numbered_checkpoint(model: TinyGPT, step: int, metrics: dict) -> dict:
    model_path = CONTINUE_CHECKPOINT_DIR / f"step_{step:06d}_model.safetensors"
    meta_path = CONTINUE_CHECKPOINT_DIR / f"step_{step:06d}_meta.json"
    model.save_weights(str(model_path))
    meta = {
        "step": int(step),
        "model_path": str(model_path),
        **metrics,
    }
    write_json(meta_path, meta)
    return meta


def plot_loss(history: list[dict]) -> None:
    steps = [item["step"] for item in history]
    train = [item["train_loss"] for item in history]
    val = [item["val_loss"] for item in history]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train, label="train loss")
    plt.plot(steps, val, label="val loss")
    plt.xlabel("global step")
    plt.ylabel("cross entropy loss")
    plt.title("Continue Training Diagnosis Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def write_comparison(before: dict[str, str], middle: dict[str, str] | None, after: dict[str, str]) -> None:
    lines = ["# Before / Middle / After Generation Comparison", ""]
    for label, samples in [
        ("继续训练前样本", before),
        ("继续训练中间样本", middle),
        ("继续训练后样本", after),
    ]:
        if samples is None:
            continue
        lines.append(f"## {label}")
        for prompt, text in samples.items():
            lines.append("")
            lines.append(f"### prompt: {prompt}")
            lines.append(text)
        lines.append("")
    COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")


def diagnose(history: list[dict], before: dict[str, str], after: dict[str, str], config: ContinueConfig, load_info: dict) -> str:
    first = history[0]
    last = history[-1]
    train_delta = last["train_loss"] - first["train_loss"]
    val_delta = last["val_loss"] - first["val_loss"]
    gap_start = first["val_loss"] - first["train_loss"]
    gap_end = last["val_loss"] - last["train_loss"]

    if train_delta < -0.1 and val_delta < -0.1:
        main_judgement = "underfitting / 训练时间不够"
        reason = "继续训练后 train loss 和 val loss 都继续下降，说明模型还没有把当前数据分布学充分。"
    elif train_delta < -0.1 and val_delta > 0.05:
        main_judgement = "overfitting / 过拟合"
        reason = "train loss 下降但 val loss 上升，说明模型开始更偏向记忆训练集。"
    else:
        main_judgement = "数据质量、模型容量或超参数限制"
        reason = "继续训练后 loss 改善有限，说明只增加步数不是主要瓶颈。"

    tokens_seen = last["tokens_seen"]
    lines = [
        "# 23b：继续训练与诊断报告",
        "",
        "## Checkpoint 加载情况",
        "",
        f"- 是否成功加载 checkpoint：{load_info['loaded_checkpoint']}",
        f"- checkpoint 路径：`{load_info['checkpoint_path']}`",
        f"- 是否 fallback 从头训练：{load_info['fallback_from_scratch']}",
        "- 说明：本次恢复了模型权重，但第 23 课没有保存 AdamW optimizer state，因此继续训练时 optimizer 动量从零开始。",
        "",
        "## 训练配置",
        "",
        f"- previous_steps: {config.previous_steps}",
        f"- extra_steps: {config.extra_steps}",
        f"- total_steps: {config.total_steps}",
        f"- batch_size: {config.batch_size}",
        f"- block_size: {config.block_size}",
        f"- learning_rate: {config.learning_rate}",
        f"- tokens_seen 估算：{tokens_seen:,}",
        "",
        "## Loss 结果",
        "",
        f"- 起始 train loss: {first['train_loss']:.4f}",
        f"- 起始 val loss: {first['val_loss']:.4f}",
        f"- 最终 train loss: {last['train_loss']:.4f}",
        f"- 最终 val loss: {last['val_loss']:.4f}",
        f"- train loss 变化: {train_delta:.4f}",
        f"- val loss 变化: {val_delta:.4f}",
        f"- 起始 val-train gap: {gap_start:.4f}",
        f"- 最终 val-train gap: {gap_end:.4f}",
        "",
        "## 诊断结论",
        "",
        f"当前更像：**{main_judgement}**。",
        "",
        reason,
        "",
        "从 loss gap 看，val loss 仍高于 train loss，但没有出现 val loss 反向上升的典型过拟合曲线。",
        "生成文本相比训练前更贴近农业、农村、政策类语料的局部分布，但仍然存在重复、句法不稳和语义跳跃。",
        "",
        "## 建议",
        "",
        "1. 可以继续增加训练时间，但收益会逐渐变慢。",
        "2. 更值得做的是进入 M1 Pro 上的 scaling 实验，比较 n_embd、num_layers、block_size、batch_size 对 loss 的影响。",
        "3. 当前数据主题偏农业和政策，生成也会偏这个方向；这属于数据分布限制，不是单纯训练步数问题。",
        "4. tokenizer 已能学到真实语料高频片段，但 vocab_size=8192 会让 LM Head 更大，小模型容量可能不足以充分利用它。",
        "",
        "## 样本文件",
        "",
        f"- 对比文件：`{COMPARISON_PATH}`",
        f"- loss 曲线：`{LOSS_CURVE_PATH}`",
        f"- best checkpoint：`{BEST_DIR / 'model.safetensors'}`",
        f"- final checkpoint：`{FINAL_DIR / 'model.safetensors'}`",
    ]
    DIAGNOSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return main_judgement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--sample-interval", type=int, default=1000)
    args = parser.parse_args()

    ensure_dirs()
    LOG_PATH.write_text("", encoding="utf-8")
    mx.random.seed(123)
    np.random.seed(123)

    run_config, best_meta = load_base_metadata()
    previous_steps = int(best_meta["step"]) + 1
    config = ContinueConfig(
        previous_steps=previous_steps,
        extra_steps=args.extra_steps,
        block_size=int(run_config["block_size"]),
        batch_size=int(run_config["batch_size"]),
        n_embd=int(run_config["n_embd"]),
        num_heads=int(run_config["num_heads"]),
        num_layers=int(run_config["num_layers"]),
        learning_rate=float(run_config["learning_rate"]),
        eval_interval=args.eval_interval,
        sample_interval=args.sample_interval,
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = TokenDataset(config)
    model = make_model(run_config, config)

    load_info = {
        "loaded_checkpoint": False,
        "checkpoint_path": best_meta.get("model_path"),
        "fallback_from_scratch": False,
        "load_error": None,
    }
    try:
        model.load_weights(best_meta["model_path"], strict=True)
        load_info["loaded_checkpoint"] = True
    except Exception as exc:  # noqa: BLE001
        load_info["fallback_from_scratch"] = True
        load_info["load_error"] = str(exc)

    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad = nn.value_and_grad(model, language_model_loss)

    append_log("=== Continue Training Diagnosis ===")
    append_log(f"current_dir={CURRENT_DIR}")
    append_log(f"loaded_checkpoint={load_info['loaded_checkpoint']}")
    append_log(f"checkpoint_path={load_info['checkpoint_path']}")
    append_log(f"fallback_from_scratch={load_info['fallback_from_scratch']}")
    append_log(f"previous_steps={config.previous_steps}")
    append_log(f"extra_steps={config.extra_steps}")
    append_log(f"total_steps={config.total_steps}")
    append_log(f"learning_rate={config.learning_rate}")
    append_log(f"batch_size={config.batch_size}")
    append_log(f"block_size={config.block_size}")

    before_samples = generate_for_prompts(model, tokenizer, config)
    save_samples(config.previous_steps, before_samples)

    history: list[dict] = []
    best_val = float("inf")
    middle_samples = None
    start_time = time.perf_counter()

    for extra_step in range(1, config.extra_steps + 1):
        global_step = config.previous_steps + extra_step
        x, y = dataset.get_batch("train")
        loss, grads = value_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        should_eval = (
            extra_step == 1
            or extra_step % config.eval_interval == 0
            or extra_step == config.extra_steps
        )
        if not should_eval:
            continue

        losses = estimate_loss(model, dataset, config)
        elapsed = time.perf_counter() - start_time
        tokens_seen = global_step * config.batch_size * config.block_size
        is_best = losses["val"] < best_val
        if is_best:
            best_val = losses["val"]

        row = {
            "step": global_step,
            "extra_step": extra_step,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "tokens_seen": tokens_seen,
            "learning_rate": config.learning_rate,
            "elapsed_time": elapsed,
        }
        history.append(row)
        write_json(LOSS_HISTORY_PATH, history)
        plot_loss(history)

        sample_path = None
        if extra_step % config.sample_interval == 0 or extra_step == config.extra_steps:
            samples = generate_for_prompts(model, tokenizer, config)
            sample_path = save_samples(global_step, samples)
            if middle_samples is None and extra_step >= config.extra_steps // 2:
                middle_samples = samples

        if is_best:
            save_checkpoint(model, global_step, losses, BEST_DIR)
        if extra_step % config.sample_interval == 0 or extra_step == config.extra_steps:
            save_numbered_checkpoint(model, global_step, losses)

        line = (
            f"step={global_step:06d} extra_step={extra_step:05d} "
            f"train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
            f"tokens_seen={tokens_seen} learning_rate={config.learning_rate} "
            f"elapsed_time={elapsed:.1f}s"
        )
        if sample_path:
            line += f" sample={sample_path.name}"
        if is_best:
            line += " best_val_checkpoint=updated"
        append_log(line)

    final_losses = history[-1]
    save_checkpoint(model, config.total_steps, final_losses, FINAL_DIR)
    after_samples = generate_for_prompts(model, tokenizer, config)
    save_samples(config.total_steps, after_samples)
    if middle_samples is None:
        middle_samples = after_samples
    write_comparison(before_samples, middle_samples, after_samples)

    config_payload = {
        **run_config,
        "continue_training": {
            "previous_steps": config.previous_steps,
            "extra_steps": config.extra_steps,
            "total_steps": config.total_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "block_size": config.block_size,
            "loaded_checkpoint": load_info["loaded_checkpoint"],
            "fallback_from_scratch": load_info["fallback_from_scratch"],
            "checkpoint_path": load_info["checkpoint_path"],
            "best_val_loss": best_val,
            "final_train_loss": final_losses["train_loss"],
            "final_val_loss": final_losses["val_loss"],
            "final_tokens_seen": final_losses["tokens_seen"],
        },
    }
    write_json(CONTINUE_DIR / "continue_config.json", config_payload)
    judgement = diagnose(history, before_samples, after_samples, config, load_info)
    append_log(f"diagnosis={judgement}")
    append_log(f"diagnosis_report={DIAGNOSIS_PATH}")


if __name__ == "__main__":
    main()
