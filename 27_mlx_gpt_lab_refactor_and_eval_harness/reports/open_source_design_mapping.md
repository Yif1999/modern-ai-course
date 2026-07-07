# 开源项目设计映射

本工程是教学型 MLX GPT Lab，不是通用大模型训练平台。它借鉴开源项目的工程思想，但刻意保留小而透明的结构。

---

## nanoGPT：借鉴最小训练闭环

借鉴点：

- `model.py` 和 `train.py` 分离。
- 配置驱动训练。
- 数据预处理和训练解耦。
- checkpoint、sample、loss curve 都属于训练 run 的产物。
- 训练脚本尽量直观，不把核心逻辑藏进复杂框架。

本工程对应实现：

```text
src/mlx_gpt_lab/model_baseline.py
src/mlx_gpt_lab/model_qwen_dense.py
src/mlx_gpt_lab/train_loop.py
scripts/train.py
```

没有照搬的部分：

- 不复刻 GPT-2 训练。
- 不接 OpenWebText。
- 不做多 GPU / DDP。

原因：

本课程设备目标是 M1 Pro / 32GB，重点是理解机制和工程边界。

---

## LitGPT：借鉴工程组织

借鉴点：

- 训练、生成、评估、benchmark 分入口。
- 多个 config 管理不同实验。
- 输出目录按 run 保存。
- 后续可以扩展 LoRA / SFT，但本节不提前实现。

本工程对应实现：

```text
configs/
scripts/train.py
scripts/generate.py
scripts/evaluate.py
scripts/benchmark.py
outputs/runs/<run_id>/
```

没有引入的复杂度：

- 不做完整 CLI 框架。
- 不做多模型 zoo。
- 不实现 LoRA / QLoRA / adapter。
- 不做 deploy server。

原因：

LitGPT 是工程化 LLM 项目，本工程是课程底座。我们只吸收“结构分层”和“配置化运行”。

---

## OLMo / SmolLM：借鉴可观察性和 recipe 记录

借鉴点：

- 每个 run 保存 `config.json`。
- 每个 run 保存 `metrics.json`。
- 训练日志使用 jsonl。
- 数据处理保存 metadata / manifest。
- loss curve 和 samples 都作为实验记录。
- 关注 tokens_seen、tokens/sec、val loss，而不是只看最终生成文本。

本工程对应实现：

```text
data/manifests/
data/processed/*/metadata.json
outputs/runs/<run_id>/config.json
outputs/runs/<run_id>/metrics.json
outputs/runs/<run_id>/training_log.jsonl
outputs/runs/<run_id>/loss_curve.png
outputs/runs/<run_id>/samples/
```

没有引入的复杂度：

- 不做真实大规模 data mix。
- 不做多阶段 pretraining。
- 不做完整 benchmark suite。

原因：

M1 Pro 上适合学习“实验记录应该长什么样”，不适合复刻 fully open 大模型训练规模。

---

## MLX-LM：借鉴 Apple Silicon 方向

借鉴点：

- 以 MLX 为核心后端。
- 目标运行设备是 Apple Silicon。
- 保留后续切换到真实模型推理、量化、LoRA 的接口意识。

本工程当前做的事：

- 从零实现教学型 Tiny GPT。
- 统一 tokenizer / dataset / model / train / generate。
- 让用户理解真实 MLX-LM 之前的底层机制。

没有做的事：

- 不加载真实 Hugging Face 模型。
- 不做量化。
- 不做 LoRA。
- 不做 MLX-LM server。

原因：

从零实现 Tiny GPT 是理解阶段；MLX-LM 是后续真实模型实践阶段。

---

## Qwen：借鉴中文 dense decoder-only 主线

借鉴点：

- 中文 / 多语种优先。
- dense decoder-only 作为默认主线。
- 现代组件使用 RoPE、RMSNorm、SwiGLU、Pre-Norm。
- 尽量少 bias。
- 预留 GQA / QK-Norm 配置思路，但本节不强制实现。

本工程对应实现：

```text
src/mlx_gpt_lab/model_qwen_dense.py
configs/qwen_dense_tiny_debug.json
configs/qwen_dense_tiny_zh_bpe_small.json
```

没有实现的 Qwen / 现代 LLM 复杂度：

- 不实现 MoE。
- 不实现完整 Qwen tokenizer / chat template。
- 不实现 GQA / QK-Norm。
- 不实现长上下文工程。

原因：

本节目标是建立工程底座，不是在一个课里堆所有现代架构。

---

## 为什么适合 M1 Pro / 32GB

本工程保持小而透明：

- 默认 smoke test 只有几十 step。
- 模型参数量在几万级别。
- 不下载新数据。
- fallback 中文语料可保证离线运行。
- 输出目录完整，便于后续课程复用。

这比直接引入大型框架更适合当前阶段。
