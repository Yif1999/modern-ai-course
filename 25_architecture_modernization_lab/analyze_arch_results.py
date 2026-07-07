from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
REPORT_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

SUMMARY_CSV_PATH = REPORT_DIR / "architecture_summary.csv"
REPORT_MD_PATH = REPORT_DIR / "architecture_comparison_report.md"
LOSS_PLOT_PATH = PLOTS_DIR / "loss_comparison.png"
SPEED_PLOT_PATH = PLOTS_DIR / "tokens_per_second_comparison.png"


def read_metrics() -> list[dict]:
    rows = []
    for path in sorted(RUNS_DIR.glob("*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) < 2:
        raise FileNotFoundError("需要 baseline_tiny_gpt 和 modern_tiny_gpt 的 metrics.json")
    return rows


def read_snippet(path: str, limit: int = 400) -> str:
    text = Path(path).read_text(encoding="utf-8").strip().replace("\n", " ")
    return text[:limit]


def write_csv(rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "parameter_count",
        "block_size",
        "batch_size",
        "n_embd",
        "num_heads",
        "num_layers",
        "tokens_seen",
        "tokens_per_second",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "overfit_gap",
        "elapsed_sec",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def plot_bar(rows: list[dict], key: str, ylabel: str, title: str, path: Path) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [row["run_name"].replace("_tiny_gpt", "") for row in rows]
    values = [row[key] for row in rows]
    plt.figure(figsize=(7, 5))
    plt.bar(names, values, color=["#4C78A8", "#F58518"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_report(rows: list[dict]) -> None:
    by_name = {row["run_name"]: row for row in rows}
    baseline = by_name["baseline_tiny_gpt"]
    modern = by_name["modern_tiny_gpt"]

    val_delta = modern["final_val_loss"] - baseline["final_val_loss"]
    speed_ratio = modern["tokens_per_second"] / max(baseline["tokens_per_second"], 1e-9)
    param_delta = modern["parameter_count"] - baseline["parameter_count"]
    param_ratio = modern["parameter_count"] / max(baseline["parameter_count"], 1)

    lines = [
        "# 第 25 课：现代 LLM 架构升级实验报告",
        "",
        "## 实验设置",
        "",
        "本实验使用同一份中文 BPE 数据、同样的训练预算，对比旧版 Tiny GPT 和现代化 Tiny GPT。",
        "",
        "baseline 使用 learned position embedding、LayerNorm、GELU FeedForward、独立 LM Head。",
        "modern 使用 RoPE、RMSNorm、SwiGLU 和 Weight Tying。",
        "",
        "## 指标汇总",
        "",
        "| run | params | train loss | val loss | best val | tokens/sec | gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {parameter_count} | {final_train_loss:.4f} | {final_val_loss:.4f} | "
            "{best_val_loss:.4f} | {tokens_per_second:.1f} | {overfit_gap:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## 结构差异",
            "",
            "### baseline_tiny_gpt",
            "",
            "- learned position embedding：位置向量直接加到 token embedding 上。",
            "- LayerNorm：先减均值，再除标准差。",
            "- GELU FeedForward：`Linear -> GELU -> Linear`。",
            "- 独立 LM Head：输出层有自己的 `n_embd -> vocab_size` 权重。",
            "",
            "### modern_tiny_gpt",
            "",
            "- RoPE：不再加 learned position embedding，而是在 attention 中旋转 Q/K。",
            "- RMSNorm：只按均方根缩放，不减均值。",
            "- SwiGLU：用门控 `SiLU(gate) * value` 增强 FFN 表达能力。",
            "- Weight Tying：LM Head 使用 token embedding table 的转置，减少输出层参数。",
            "",
            "## 结果判断",
            "",
            f"- modern 相对 baseline 的 val loss 变化：`{val_delta:+.4f}`。",
            f"- modern tokens/sec 是 baseline 的 `{speed_ratio:.2f}` 倍。",
            f"- modern 参数量变化：`{param_delta:+d}`，参数量比例 `{param_ratio:.2f}`。",
        ]
    )
    if val_delta < 0:
        lines.append("- 在本次短训练里，modern 的 val loss 更低。")
    else:
        lines.append("- 在本次短训练里，modern 的 val loss 没有低于 baseline。")

    lines.extend(
        [
            "",
            "## 如何理解这个结果？",
            "",
            "现代组件不是魔法。它们通常在更大模型、更长训练、更严格调参下更明显。",
            "本实验规模很小，所以如果 modern 没有显著胜出，也不能说明 RoPE/RMSNorm/SwiGLU 没有价值。",
            "",
            "Weight Tying 的参数效率最直观：它减少了独立 LM Head 的大矩阵。",
            "RMSNorm 和 RoPE 更偏现代 decoder-only LLM 的稳定性和位置建模习惯。",
            "SwiGLU 更偏表达能力，但参数和计算也会增加。",
            "",
            "## 生成样本摘录",
            "",
            "### baseline_tiny_gpt",
            "",
            read_snippet(baseline["final_generated_text_path"]),
            "",
            "### modern_tiny_gpt",
            "",
            read_snippet(modern["final_generated_text_path"]),
            "",
            "## 下一步",
            "",
            "可以进入 `26_open_training_recipe_review`。",
            "下一步不应只看单个组件，而要把数据、训练预算、学习率、warmup、归一化、位置编码和采样策略放在完整 recipe 里审视。",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_metrics()
    write_csv(rows)
    plot_bar(rows, "final_val_loss", "final val loss", "Architecture Val Loss Comparison", LOSS_PLOT_PATH)
    plot_bar(rows, "tokens_per_second", "tokens/sec", "Architecture Speed Comparison", SPEED_PLOT_PATH)
    write_report(rows)
    print("=== Architecture results analyzed ===")
    print("report:", REPORT_MD_PATH)
    print("summary:", SUMMARY_CSV_PATH)
    print("plots:", LOSS_PLOT_PATH, SPEED_PLOT_PATH)


if __name__ == "__main__":
    main()
