# MLX GPT Lab 改进计划

这一份计划基于 nanoGPT、LitGPT、OLMo、SmolLM3、Qwen、DeepSeek 和 MLX-LM 的阅读结论，目标是把我们当前课程里的零散实验，整理成一个可复用、可观察、可扩展的本地 MLX GPT Lab。

---

## 当前问题

我们目前已经有很多课程目录：

```text
15_tiny_gpt_training
16_tiny_gpt_sampling
17_kv_cache_intro
19_tiny_gpt_training_pipeline
20_chinese_open_text_pretraining_dataset
21_chinese_bpe_tokenizer_intro
22_train_tiny_gpt_with_chinese_bpe
23_chinese_open_dataset_pretraining_run
24_chinese_gpt_scaling_on_m1_pro
25_architecture_modernization_lab
```

这些实验完成了概念学习，但还存在工程问题：

- 模型定义重复。
- tokenizer / dataset / train / generate 分散在多个目录。
- 配置不统一。
- eval 没有形成标准入口。
- runs 的命名和 metadata 还不够统一。
- checkpoint 加载和生成脚本可以继续标准化。
- 报告和图表可以更自动化。

---

## 推荐目录结构

建议后续建立一个统一项目目录，例如：

```text
mlx_gpt_lab/
├── configs/
│   ├── pretrain_tiny_zh.yaml
│   ├── pretrain_modern_zh.yaml
│   ├── lora_qwen_small.yaml
│   └── eval_smoke.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── manifests/
│   └── README.md
├── tokenizer/
│   ├── train_bpe.py
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── reports/
├── src/
│   ├── mlx_gpt_lab/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   ├── model_baseline.py
│   │   ├── model_modern.py
│   │   ├── train_loop.py
│   │   ├── generate.py
│   │   ├── checkpoint.py
│   │   ├── sampling.py
│   │   ├── evals.py
│   │   └── utils.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── export_report.py
├── evals/
│   ├── prompts_zh.jsonl
│   ├── tiny_language_eval.py
│   ├── perplexity_eval.py
│   └── generation_quality_check.py
├── benchmarks/
│   ├── tokens_per_second.py
│   ├── memory_usage.py
│   └── kv_cache_benchmark.py
├── checkpoints/
│   └── README.md
├── outputs/
│   ├── runs/
│   │   └── 2026-xx-xx_run_name/
│   │       ├── config.json
│   │       ├── metrics.json
│   │       ├── training_log.jsonl
│   │       ├── loss_curve.png
│   │       ├── samples/
│   │       ├── checkpoints/
│   │       └── report.md
│   └── reports/
├── reports/
│   ├── experiment_index.md
│   └── architecture_notes.md
└── README.md
```

---

## configs/

借鉴 LitGPT 和 OLMo，所有实验都应该由配置驱动。

配置至少包含：

- 数据路径。
- tokenizer 路径。
- block_size。
- batch_size。
- n_embd。
- num_heads。
- num_layers。
- learning_rate。
- max_iters。
- eval_interval。
- seed。
- checkpoint 策略。
- 采样策略。

建议先用 JSON 或 YAML。为了简单，课程下一阶段可以先使用 JSON。

---

## data/

借鉴 OLMo 的数据透明度，数据目录不要只放 token ids，还要放 manifest。

推荐：

```text
data/raw/ 原始文本或 jsonl 样本
data/interim/ 清洗后的中间文本
data/processed/ train_tokens.npy / val_tokens.npy
data/manifests/ dataset_manifest.json
```

每个数据集都应该记录：

- 数据来源。
- 抽样时间。
- 抽样上限。
- 原始字符数。
- 清洗后字符数。
- token 数。
- train / val split。
- 中文比例。
- 去重数量。
- tokenizer 版本。

---

## outputs/runs/

借鉴 LitGPT / OLMo 的 run 管理，每次训练都应该有独立目录：

```text
outputs/runs/<timestamp>_<run_name>/
```

每个 run 至少保存：

- `config.json`
- `metrics.json`
- `training_log.jsonl`
- `loss_curve.png`
- `samples/`
- `checkpoints/`
- `report.md`

这会让后续 scaling、消融实验和复盘更容易。

---

## src/

借鉴 nanoGPT 的清晰性，但避免复制粘贴太多模型代码。

推荐拆分：

- `model_baseline.py`：朴素 Tiny GPT。
- `model_modern.py`：RoPE / RMSNorm / SwiGLU / weight tying。
- `dataset.py`：batch 采样。
- `tokenizer.py`：字符级 / BPE 包装。
- `train_loop.py`：通用训练循环。
- `checkpoint.py`：保存 / 加载。
- `generate.py`：通用生成。
- `sampling.py`：temperature / top-k / top-p。
- `evals.py`：loss / perplexity / prompt eval。

---

## evals/

后续必须补上评估层。只看 loss 和生成样本不够。

M1 Pro 本地可做的轻量评估：

- 固定 prompt 生成对比。
- held-out validation perplexity。
- 中文标点和重复率统计。
- 简单 QA prompt 人工检查。
- 训练样本泄漏检查。
- 长上下文回忆测试。

先不要做 MMLU / C-Eval 全量评测，但可以保留接口。

---

## benchmarks/

本地设备非常需要速度和内存记录。

建议记录：

- tokens/sec。
- peak memory 或近似内存占用。
- batch_size / block_size 对速度影响。
- KV Cache 有无对推理速度影响。
- 不同采样策略速度。

---

## checkpoints/

checkpoint 命名建议：

```text
step_0001000/
step_0002000/
best_val/
final/
```

每个 checkpoint 目录应包含：

- 模型参数。
- config。
- tokenizer 引用。
- 当前 step。
- best val loss。

---

## tokenizer/

后续中文主线应该稳定使用 BPE tokenizer。

推荐记录：

- tokenizer 训练语料。
- vocab_size。
- special tokens。
- encode / decode 样例。
- 字符级 vs BPE token 数对比。
- tokenizer 文件 hash 或版本号。

---

## scripts/

命令入口保持简单：

```bash
python scripts/prepare_data.py --config configs/pretrain_tiny_zh.json
python scripts/train.py --config configs/pretrain_tiny_zh.json
python scripts/generate.py --run outputs/runs/xxx --prompt "人工智能"
python scripts/evaluate.py --run outputs/runs/xxx
python scripts/benchmark.py --run outputs/runs/xxx
```

---

## reports/

每次阶段结束后生成机器可读和人类可读报告：

- `metrics.json`
- `training_log.jsonl`
- `report.md`
- `experiment_index.md`

这部分直接借鉴 OLMo 的透明实验理念。

---

## 分阶段改进路线

### Phase 1：重构，不改模型

目标：把 23-25 课代码合并成标准项目结构。

完成标准：

- 一个 `train.py` 支持 baseline / modern。
- 一个 `generate.py` 支持 checkpoint 生成。
- 一个 `prepare_data.py` 支持中文 BPE 数据。
- 每次 run 自动生成完整输出目录。

### Phase 2：补 eval harness

目标：不只看 loss。

完成标准：

- 固定 prompt 评估。
- validation perplexity。
- 重复率统计。
- tokens/sec benchmark。

### Phase 3：接 MLX-LM

目标：从自写 Tiny GPT 过渡到真实模型。

完成标准：

- 使用 MLX-LM 加载 Qwen 小模型或其他中文模型。
- 运行本地 generate。
- 准备 LoRA 数据。
- 做最小 LoRA 微调。

### Phase 4：后训练入门

目标：进入 SFT / preference / reasoning 数据，而不是继续盲目预训练。

完成标准：

- 指令数据格式。
- chat template。
- SFT loss。
- LoRA checkpoint。
- 简单人工评估。

---

## 最重要的设计原则

1. 配置和代码分离。
2. 数据和 tokenizer 可追溯。
3. 每次 run 独立保存。
4. 训练、生成、评估、benchmark 分离。
5. 小模型实验也要保留真实工程记录。
6. MLX 自写模型用于理解，MLX-LM 用于真实模型实践。
