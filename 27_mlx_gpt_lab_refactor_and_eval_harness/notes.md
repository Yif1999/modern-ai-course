# 第 27 课：MLX GPT Lab 工程化底座

## 这一课解决什么问题？

前面课程已经实现了很多分散实验：

- Tiny GPT 训练。
- sampling。
- KV Cache。
- 中文数据管线。
- BPE tokenizer。
- scaling。
- 现代架构。

但这些实验分散在不同课程目录中，模型、训练、生成、评估逻辑有重复。

这一课的目标是把它们整理成一个长期可维护的教学型 MLX GPT Lab。

---

## 本工程不是什么？

它不是：

- LLaMA-Factory。
- Axolotl。
- LitGPT。
- MLX-LM。
- 通用训练平台。

它是：

```text
教学型 MLX GPT Lab
```

核心目标是帮助我们在 Apple Silicon / MLX 上理解 GPT 训练机制，并为后续中文 continued pretraining、toy SFT、persona 实验打底。

---

## 为什么只保留两个模型 profile？

本工程只保留：

```text
baseline_debug
qwen_dense_tiny
```

`baseline_debug` 用于快速排错。

它结构简单：

```text
learned position embedding
LayerNorm
GELU FFN
普通 causal MHA
```

`qwen_dense_tiny` 是后续主线模型。

它结构更接近现代 dense decoder-only LLM：

```text
RoPE
RMSNorm
SwiGLU
Pre-Norm
causal self-attention
optional weight tying
```

不同时实现 llama_like / smollm_like / qwen_like，是为了避免模型动物园式复杂度。当前最重要的是让数据、训练、生成、评估、benchmark 链路稳定。

---

## 工程目录

核心结构：

```text
configs/
data/
src/mlx_gpt_lab/
scripts/
evals/
outputs/runs/
reports/
```

其中：

- `configs/`：实验配置。
- `data/`：raw text、processed tokens、metadata、manifest。
- `src/mlx_gpt_lab/`：可复用 Python 包。
- `scripts/`：命令入口。
- `evals/`：评估 prompt。
- `outputs/runs/`：每次训练的完整输出。
- `reports/`：设计报告和复盘。

---

## 训练链路

命令：

```bash
python scripts/train.py --config configs/baseline_debug.json
```

实际流程：

```text
读取 config
↓
准备 raw text / tokenizer / train tokens / val tokens
↓
创建 model
↓
创建 optimizer
↓
训练 loop
↓
定期 eval
↓
定期 generate sample
↓
保存 checkpoint / loss curve / metrics
```

每个 run 保存到：

```text
outputs/runs/<timestamp>_<run_name>/
```

---

## 生成链路

命令：

```bash
python scripts/generate.py --config configs/qwen_dense_tiny_debug.json --prompt "人工智能"
```

流程：

```text
找到最新 run
↓
加载 run config
↓
加载 tokenizer
↓
加载 checkpoint
↓
根据 prompt 生成
```

---

## Eval harness

命令：

```bash
python scripts/evaluate.py --config configs/eval_smoke.json --train-config configs/qwen_dense_tiny_debug.json
```

当前 eval 很轻量，只检查：

- 生成是否为空。
- 平均生成长度。
- 重复字符比例。
- 重复 3-gram 比例。
- 是否有明显乱码标记。

它不是质量评测，只是 smoke eval。

---

## Benchmark

命令：

```bash
python scripts/benchmark.py --config configs/qwen_dense_tiny_debug.json
```

当前 benchmark 记录：

- 生成 tokens/sec。
- 生成耗时。
- 参数量估算。
- 使用的 checkpoint。

它是教学级 benchmark，不是严格性能测试。

---

## Smoke test 观察

baseline_debug 和 qwen_dense_tiny_debug 都完成了 40 step 训练。

两者 loss 都下降，说明：

```text
数据 -> tokenizer -> batch -> model -> loss -> optimizer -> checkpoint -> generate
```

这条主链路已经打通。

生成文本仍然很弱，这是预期的。原因是：

- fallback 数据很小。
- 训练只有 40 step。
- 字符级 tokenizer 粒度很细。
- 模型参数量很小。

本节重点不是生成质量，而是工程底座。

---

## 下一步

第 28 课建议接入：

```text
中文趣味语料 continued pretraining
```

需要做：

1. 替换 fallback 数据。
2. 使用 BPE tokenizer。
3. 跑 qwen_dense_tiny 主线模型。
4. 增强 eval prompts。
5. 对比训练前后风格变化。
