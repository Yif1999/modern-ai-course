# 第 27 课重构总结

## 已完成

### 工程目录

已创建：

```text
configs/
data/raw/
data/processed/
data/manifests/
data/reports/
src/mlx_gpt_lab/
scripts/
evals/
outputs/runs/
reports/
```

### 配置系统

已创建：

```text
configs/baseline_debug.json
configs/qwen_dense_tiny_debug.json
configs/qwen_dense_tiny_zh_bpe_small.json
configs/zh_fun_pretrain_placeholder.json
configs/eval_smoke.json
```

支持字段：

- `model_type`
- `tokenizer_type`
- `data_path`
- `tokenizer_path`
- `block_size`
- `batch_size`
- `n_embd`
- `num_heads`
- `num_layers`
- `learning_rate`
- `max_iters`
- `eval_interval`
- `sample_interval`
- `seed`
- `run_name`

### 数据和 tokenizer

已实现：

- fallback 中文小语料。
- 字符级 tokenizer。
- BPE tokenizer 支持。
- train / val token ids。
- metadata。
- manifest。
- batch sampling。

### 模型

已实现：

- `baseline_debug`
- `qwen_dense_tiny`

`qwen_dense_tiny` 包含：

- RoPE
- RMSNorm
- SwiGLU
- Pre-Norm
- optional weight tying

### 训练

已实现：

- 读取 config。
- 准备数据。
- 创建模型。
- 训练。
- 定期 eval。
- 定期生成样本。
- 保存 checkpoint。
- 保存 loss curve。
- 保存 training log。
- 保存 config 副本。

### 生成

已实现：

- 从 run 目录加载 config。
- 加载 tokenizer。
- 加载 checkpoint。
- 根据 prompt 生成。
- 支持 temperature / top-k / top-p。

### Eval harness

已实现：

- 读取 prompt jsonl。
- 生成结果。
- 保存 `eval_results.jsonl`。
- 计算简单指标：
  - 平均生成长度。
  - 空输出数量。
  - 重复字符比例。
  - 重复 3-gram 比例。
  - 明显乱码标记。

### Benchmark

已实现：

- 单次生成耗时。
- tokens/sec。
- 参数量估算。
- benchmark 输出保存。

---

## Smoke test 结果

### baseline_debug

输出目录：

```text
outputs/runs/20260612_104206_baseline_debug/
```

结果：

```text
train loss: 1.9703
val loss: 1.8672
tokens_seen: 5120
tokens/sec: 13968.21
```

生成文件：

- `config.json`
- `training_log.jsonl`
- `loss_curve.png`
- `samples/`
- `checkpoints/`
- `metrics.json`
- `final_generated_text.txt`

### qwen_dense_tiny_debug

输出目录：

```text
outputs/runs/20260612_104212_qwen_dense_tiny_debug/
```

结果：

```text
train loss: 1.8429
val loss: 1.6612
tokens_seen: 5120
tokens/sec: 13764.32
```

生成文件：

- `config.json`
- `training_log.jsonl`
- `loss_curve.png`
- `samples/`
- `checkpoints/`
- `metrics.json`
- `final_generated_text.txt`
- `evals/eval_results.jsonl`
- `evals/eval_metrics.json`
- `benchmarks/benchmark.json`

---

## 未完成 / 限制

### checkpoint

当前保存模型参数 checkpoint，可以加载做生成、eval、benchmark。

暂未保存 optimizer state。

原因：

本节重点是工程底座和 smoke test。继续训练恢复 optimizer state 会增加实现复杂度，后续可以补。

### BPE smoke test

代码支持 BPE tokenizer，并且已经用 `configs/qwen_dense_tiny_zh_bpe_small.json` 完成数据准备 smoke test：

```text
data/processed/qwen_dense_tiny_zh_bpe_small/train_tokens.npy
data/processed/qwen_dense_tiny_zh_bpe_small/val_tokens.npy
data/processed/qwen_dense_tiny_zh_bpe_small/tokenizer.json
vocab_size: 512
```

但本节两个训练 smoke test 仍使用字符级 tokenizer。

原因：

字符级更稳定，能保证离线快速跑通。BPE 配置已经准备在：

```text
configs/qwen_dense_tiny_zh_bpe_small.json
```

后续中文趣味语料课可以直接使用。

### eval 指标仍然很粗

当前 eval 只做：

- 长度。
- 重复率。
- 空输出。
- 简单乱码检测。

还没有：

- perplexity eval。
- 固定答案任务。
- 中文 QA。
- 格式遵循评估。
- LLM-as-judge。

### benchmark 仍然是教学级

当前只测单次生成 tokens/sec。

还没有：

- 多轮平均。
- warmup。
- 内存统计。
- batch 推理。
- KV Cache benchmark。

---

## 技术债

1. checkpoint 需要补 optimizer state。
2. eval harness 需要加入 validation perplexity。
3. BPE tokenizer 需要在中文趣味语料上正式 smoke test。
4. 配置系统目前是 JSON，后续可以考虑 YAML。
5. scripts 里有少量重复 CLI path 解析逻辑，可以抽成公共工具。
6. benchmark 需要更稳定的重复测量。
7. 数据 manifest 还比较简单，后续应加入 hash、来源、清洗规则和 tokenizer 版本。

---

## 后续如何补

第 28 课建议进入：

```text
中文趣味语料 continued pretraining + 更完整 eval harness
```

需要改的地方：

1. 替换 `data/raw/fallback_zh.txt` 为趣味中文语料。
2. 使用 `qwen_dense_tiny_zh_bpe_small.json`。
3. 跑 BPE prepare。
4. 增加 fun_style eval prompts。
5. 对比训练前后生成风格变化。
