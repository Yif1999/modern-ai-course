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

SUMMARY_CSV_PATH = REPORT_DIR / "scaling_summary.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "scaling_summary.json"
REPORT_MD_PATH = REPORT_DIR / "scaling_report.md"
VAL_LOSS_PLOT_PATH = PLOTS_DIR / "val_loss_comparison.png"
TOKENS_SEC_PLOT_PATH = PLOTS_DIR / "tokens_per_second_comparison.png"
LOSS_VS_TOKENS_PLOT_PATH = PLOTS_DIR / "final_loss_vs_tokens_seen.png"


def read_metrics() -> list[dict]:
    rows = []
    for path in sorted(RUNS_DIR.glob("*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"没有找到 run metrics: {RUNS_DIR}/*/metrics.json")
    return rows


def read_sample_snippet(run_name: str, limit: int = 240) -> str:
    path = RUNS_DIR / run_name / "final_generated_text.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip().replace("\n", " ")
    return text[:limit]


def write_summary_files(rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "dataset_scale",
        "dataset_actual_chars",
        "block_size",
        "batch_size",
        "n_embd",
        "num_layers",
        "num_heads",
        "max_iters",
        "tokens_seen",
        "tokens_per_second",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "best_step",
        "overfit_gap",
        "elapsed_sec",
    ]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_bars(rows: list[dict], key: str, ylabel: str, title: str, path: Path) -> None:
    names = [row["run_name"].replace("run_", "") for row in rows]
    values = [row[key] for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(rows)])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_vs_tokens(rows: list[dict]) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        plt.scatter(row["tokens_seen"], row["final_val_loss"], s=90, label=row["run_name"])
        plt.text(row["tokens_seen"], row["final_val_loss"], " " + row["run_name"].replace("run_", ""), fontsize=8)
    plt.xlabel("tokens seen")
    plt.ylabel("final val loss")
    plt.title("Final Val Loss vs Tokens Seen")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_VS_TOKENS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def trend_sentence(rows: list[dict]) -> dict[str, str]:
    by_name = {row["run_name"]: row for row in rows}
    out = {}
    baseline = by_name.get("run_a_baseline")
    if baseline and "run_b_more_data" in by_name:
        b = by_name["run_b_more_data"]
        delta = b["final_val_loss"] - baseline["final_val_loss"]
        out["more_data"] = (
            f"run_b_more_data 比 baseline 的 final val loss 变化 {delta:+.4f}。"
            + ("更多数据在本次短训练中带来下降。" if delta < 0 else "更多数据在本次短训练中没有立刻带来下降。")
        )
    if baseline and "run_c_larger_model" in by_name:
        c = by_name["run_c_larger_model"]
        delta = c["final_val_loss"] - baseline["final_val_loss"]
        speed = c["tokens_per_second"] / max(baseline["tokens_per_second"], 1e-9)
        out["larger_model"] = (
            f"run_c_larger_model 比 baseline 的 final val loss 变化 {delta:+.4f}，"
            f"tokens/sec 约为 baseline 的 {speed:.2f} 倍。"
        )
    if baseline and "run_d_longer_context" in by_name:
        d = by_name["run_d_longer_context"]
        delta = d["final_val_loss"] - baseline["final_val_loss"]
        speed = d["tokens_per_second"] / max(baseline["tokens_per_second"], 1e-9)
        out["longer_context"] = (
            f"run_d_longer_context 比 baseline 的 final val loss 变化 {delta:+.4f}，"
            f"tokens/sec 约为 baseline 的 {speed:.2f} 倍。"
        )
    return out


def write_markdown_report(rows: list[dict]) -> None:
    best_val = min(rows, key=lambda row: row["final_val_loss"])
    fastest = max(rows, key=lambda row: row["tokens_per_second"])
    best_tradeoff = min(rows, key=lambda row: row["final_val_loss"] / max(row["tokens_per_second"], 1e-9))
    trends = trend_sentence(rows)

    lines = [
        "# 第 24 课 Scaling 实验报告",
        "",
        "## 实验说明",
        "",
        "本实验在 M1 Pro / 32GB 统一内存设备上运行小型中文 Tiny GPT scaling 对比。",
        "本次优先复用第 23 课已经抽样好的真实中文开源语料和 BPE tokenizer，没有为了第 24 课额外扩大网络下载。",
        "small 使用本地语料的 100 万字符切片，medium 使用第 23 课本地完整语料。",
        "",
        "## 指标汇总",
        "",
        "| run | data | block | embd | layers | tokens_seen | tokens/sec | train loss | val loss | best val | gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {dataset_scale} | {block_size} | {n_embd} | {num_layers} | "
            "{tokens_seen} | {tokens_per_second:.1f} | {final_train_loss:.4f} | "
            "{final_val_loss:.4f} | {best_val_loss:.4f} | {overfit_gap:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 对比观察",
            "",
            f"- 最低 final val loss: `{best_val['run_name']}`，val loss = {best_val['final_val_loss']:.4f}。",
            f"- 最高 tokens/sec: `{fastest['run_name']}`，tokens/sec = {fastest['tokens_per_second']:.1f}。",
            f"- 粗略性价比最优: `{best_tradeoff['run_name']}`，它在 loss 和速度之间更均衡。",
        ]
    )
    for sentence in trends.values():
        lines.append(f"- {sentence}")

    lines.extend(
        [
            "",
            "## 生成样本摘录",
            "",
        ]
    )
    for row in rows:
        lines.append(f"### {row['run_name']}")
        lines.append("")
        lines.append(read_sample_snippet(row["run_name"]))
        lines.append("")

    lines.extend(
        [
            "## 结论",
            "",
            "这不是严格意义上的 scaling law，因为数据规模、模型规模和训练 token 数都很小，也没有对每组配置进行充分收敛训练。",
            "它的价值在于建立工程直觉：同样的 Tiny GPT 结构下，数据量、模型容量、上下文长度和训练 token 数都会改变 loss、速度和生成样本。",
            "后续如果继续扩大，应优先在可控范围内增加训练 token 数，并做 n_embd / num_layers 的小网格对比。",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_metrics()
    write_summary_files(rows)
    plot_bars(rows, "final_val_loss", "final val loss", "Final Val Loss Comparison", VAL_LOSS_PLOT_PATH)
    plot_bars(rows, "tokens_per_second", "tokens/sec", "Training Speed Comparison", TOKENS_SEC_PLOT_PATH)
    plot_loss_vs_tokens(rows)
    write_markdown_report(rows)

    print("=== Scaling Results Analyzed ===")
    print("summary csv:", SUMMARY_CSV_PATH)
    print("summary json:", SUMMARY_JSON_PATH)
    print("report:", REPORT_MD_PATH)
    print("plots:", VAL_LOSS_PLOT_PATH, TOKENS_SEC_PLOT_PATH, LOSS_VS_TOKENS_PLOT_PATH)


if __name__ == "__main__":
    main()
