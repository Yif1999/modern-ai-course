from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
LAB27_DIR = PROJECT_DIR / "27_mlx_gpt_lab_refactor_and_eval_harness"
LESSON28_DIR = PROJECT_DIR / "28_chinese_fun_corpus_pipeline"

sys.path.insert(0, str(LAB27_DIR / "src"))

from mlx_gpt_lab.train_loop import train_from_config  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_course_inputs() -> None:
    """Copy lesson 28 artifacts into lesson 29 so this lesson is self-contained."""
    copies = [
        (LESSON28_DIR / "data/processed/fun_corpus.txt", CURRENT_DIR / "data/raw/fun_corpus.txt"),
        (LESSON28_DIR / "data/raw/fun_corpus_raw.jsonl", CURRENT_DIR / "data/raw/fun_corpus_raw.jsonl"),
        (LESSON28_DIR / "data/processed/train_tokens.npy", CURRENT_DIR / "data/processed/train_tokens.npy"),
        (LESSON28_DIR / "data/processed/val_tokens.npy", CURRENT_DIR / "data/processed/val_tokens.npy"),
        (LESSON28_DIR / "data/metadata/train_val_metadata.json", CURRENT_DIR / "data/metadata/train_val_metadata_from_28.json"),
        (LESSON28_DIR / "data/metadata/lab_tokenizer_metadata.json", CURRENT_DIR / "data/metadata/lab_tokenizer_metadata_from_28.json"),
        (LESSON28_DIR / "data/metadata/qwen_token_stats.json", CURRENT_DIR / "data/metadata/qwen_token_stats_from_28.json"),
        (LESSON28_DIR / "data/metadata/source_stats.json", CURRENT_DIR / "data/metadata/source_stats_from_28.json"),
    ]
    for src, dst in copies:
        if not src.exists():
            raise FileNotFoundError(f"缺少第 28 课产物: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    lab_meta = load_json(LESSON28_DIR / "data/metadata/lab_tokenizer_metadata.json")
    tokenizer_src = Path(lab_meta["tokenizer_path"])
    tokenizer_dst = CURRENT_DIR / "data/metadata/lab_bpe_tokenizer.json"
    tokenizer_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_src, tokenizer_dst)


def collect_unk_examples(tokenizer: Tokenizer, records: list[dict], max_examples: int = 20) -> tuple[Counter, list[dict]]:
    unk_id = tokenizer.token_to_id("<unk>")
    counter: Counter[str] = Counter()
    examples: list[dict] = []
    if unk_id is None:
        return counter, examples

    for record in records:
        text = record["text"]
        encoding = tokenizer.encode(text)
        for token_id, token, offset in zip(encoding.ids, encoding.tokens, encoding.offsets):
            if token_id != unk_id and token != "<unk>":
                continue
            start, end = offset
            piece = text[start:end] if end > start else "<empty-offset>"
            if not piece:
                piece = "<empty-offset>"
            counter[piece] += 1
            if len(examples) < max_examples:
                context = text[max(0, start - 30) : min(len(text), end + 30)]
                examples.append(
                    {
                        "source_name": record.get("source_name"),
                        "source_type": record.get("source_type"),
                        "piece": piece,
                        "context": context,
                    }
                )
    return counter, examples


def write_tokenizer_coverage_report() -> dict:
    raw_path = CURRENT_DIR / "data/raw/fun_corpus_raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lab_meta = load_json(CURRENT_DIR / "data/metadata/lab_tokenizer_metadata_from_28.json")
    qwen_stats = load_json(CURRENT_DIR / "data/metadata/qwen_token_stats_from_28.json")
    tokenizer = Tokenizer.from_file(str(CURRENT_DIR / "data/metadata/lab_bpe_tokenizer.json"))

    total_sample_tokens = int(lab_meta.get("sample_token_total", lab_meta.get("token_total", 0)))
    unk_count = int(lab_meta["unk_token_count"])
    unk_ratio = float(lab_meta["unk_token_ratio"])
    counter, examples = collect_unk_examples(tokenizer, records)

    top_unknown = [{"piece": piece, "count": count} for piece, count in counter.most_common(40)]
    coverage = {
        "lab_tokenizer_path": str(CURRENT_DIR / "data/metadata/lab_bpe_tokenizer.json"),
        "lab_vocab_size": lab_meta["vocab_size"],
        "sample_count": lab_meta["sample_count"],
        "sample_token_total": total_sample_tokens,
        "corpus_token_total": lab_meta.get("corpus_token_total", lab_meta.get("token_total")),
        "unk_token_count": unk_count,
        "unk_token_ratio": unk_ratio,
        "qwen_tokenizer_name": qwen_stats.get("actual_tokenizer_name"),
        "qwen_vocab_size": qwen_stats.get("vocab_size"),
        "qwen_token_total": qwen_stats.get("qwen_token_total"),
        "qwen_over_512": qwen_stats.get("over_512"),
        "qwen_over_1024": qwen_stats.get("over_1024"),
        "qwen_over_2048": qwen_stats.get("over_2048"),
        "top_unknown_pieces": top_unknown,
        "unknown_examples": examples,
    }
    write_json(CURRENT_DIR / "outputs/reports/tokenizer_coverage_stats.json", coverage)

    lines = [
        "# Tokenizer 覆盖率诊断报告",
        "",
        "## lab tokenizer",
        "",
        f"- tokenizer：`{coverage['lab_tokenizer_path']}`",
        f"- vocab_size：{coverage['lab_vocab_size']}",
        f"- 样本数：{coverage['sample_count']}",
        f"- 样本 token 总数：{coverage['sample_token_total']}",
        f"- 拼接 corpus token 总数：{coverage['corpus_token_total']}",
        f"- `<unk>` token 数：{coverage['unk_token_count']}",
        f"- `<unk>` token 比例：{coverage['unk_token_ratio']:.4%}",
        "",
        "## Qwen tokenizer 兼容统计",
        "",
        f"- tokenizer：`{coverage['qwen_tokenizer_name']}`",
        f"- vocab_size：{coverage['qwen_vocab_size']}",
        f"- Qwen token 总数：{coverage['qwen_token_total']}",
        f"- 超过 512 tokens 的样本数：{coverage['qwen_over_512']}",
        f"- 超过 1024 tokens 的样本数：{coverage['qwen_over_1024']}",
        f"- 超过 2048 tokens 的样本数：{coverage['qwen_over_2048']}",
        "",
        "## 高频 `<unk>` 片段",
        "",
        "| 片段 | 次数 |",
        "|---|---:|",
    ]
    for item in top_unknown:
        piece = item["piece"].replace("\n", "\\n")
        lines.append(f"| `{piece}` | {item['count']} |")

    lines.extend(["", "## `<unk>` 上下文示例", ""])
    for i, item in enumerate(examples, 1):
        lines.extend(
            [
                f"### 示例 {i}",
                "",
                f"- 来源：{item['source_type']} / {item['source_name']}",
                f"- 未知片段：`{item['piece']}`",
                f"- 上下文：{item['context']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            "- `<unk>` 比例来自 tokenizer 覆盖率，训练本身不会改变 tokenizer 覆盖率。",
            "- 如果生成样本出现信息丢失、怪字符或网络梗表达弱，当前 lab tokenizer 可能是关键因素。",
            "- 下一步可以考虑用通用中文语料和趣味语料混合重训 lab BPE tokenizer，或者增大 vocab_size。",
        ]
    )
    report_path = CURRENT_DIR / "outputs/reports/tokenizer_coverage_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return coverage


def copy_run_samples_to_lesson_outputs(run_dir: Path) -> None:
    target = CURRENT_DIR / "outputs/samples" / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for sample_path in sorted((run_dir / "samples").glob("*.txt")):
        shutil.copy2(sample_path, target / sample_path.name)
    final_path = run_dir / "final_generated_text.txt"
    if final_path.exists():
        shutil.copy2(final_path, target / "final_generated_text.txt")


def write_dashboard_note(run_dirs: list[Path]) -> None:
    lines = [
        "# Dashboard 读取说明",
        "",
        "第 27 课 Dashboard 默认读取第 27 课的 `outputs/runs`。",
        "",
        "要查看第 29 课 run，可以用环境变量启动后端：",
        "",
        "```bash",
        "cd /Volumes/T7/Dev/ai-lab/27_mlx_gpt_lab_refactor_and_eval_harness/web_backend",
        "source /Volumes/T7/Dev/ai-lab/.venv/bin/activate",
        "MLX_GPT_LAB_RUNS_DIR=/Volumes/T7/Dev/ai-lab/29_fun_corpus_continued_pretraining_smoke/outputs/runs uvicorn app:app --host 0.0.0.0 --port 8765",
        "```",
        "",
        "本次生成的 run：",
        "",
    ]
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir.name}`")
    (CURRENT_DIR / "outputs/reports/dashboard_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="第 29 课：趣味语料 continued pretraining smoke")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args()

    ensure_course_inputs()
    coverage = write_tokenizer_coverage_report()
    print("Tokenizer coverage report written.")
    print(f"lab <unk> ratio: {coverage['unk_token_ratio']:.4%}")

    run_dirs: list[Path] = []
    if not args.skip_baseline:
        run_dir = train_from_config(CURRENT_DIR / "configs/baseline_fun_smoke.json", CURRENT_DIR)
        copy_run_samples_to_lesson_outputs(run_dir)
        run_dirs.append(run_dir)
    if not args.skip_qwen:
        run_dir = train_from_config(CURRENT_DIR / "configs/qwen_dense_tiny_fun_smoke.json", CURRENT_DIR)
        copy_run_samples_to_lesson_outputs(run_dir)
        run_dirs.append(run_dir)

    write_dashboard_note(run_dirs)
    print("Completed runs:")
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
