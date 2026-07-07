# 架构决策记录

## 决策 1：只保留两类模型 profile

本工程只保留：

```text
baseline_debug
qwen_dense_tiny
```

不同时实现：

```text
llama_like
smollm_like
qwen_like
```

原因：

1. 课程目标是建立工程底座，不是做模型动物园。
2. 多套类似架构会增加代码重复。
3. 用户真正需要的是一个稳定主线，便于后续数据、训练、eval 和 benchmark 复用。
4. Qwen-like dense decoder-only 更贴近中文 / 多语种后续路线。

---

## baseline_debug

用途：

- smoke test。
- 回归测试。
- 教学对照。
- 快速确认训练链路是否正常。

结构：

```text
token embedding
+ learned position embedding
Transformer Block × N
  LayerNorm
  causal MHA
  residual
  LayerNorm
  GELU FFN
  residual
final LayerNorm
LM Head
```

为什么保留：

它结构朴素，适合排错。只要 baseline 能跑，就能判断数据、tokenizer、batch、loss、optimizer、checkpoint 的基本链路是否正常。

---

## qwen_dense_tiny

用途：

- 后续默认主模型。
- 中文 BPE Tiny GPT 训练。
- continued pretraining。
- toy SFT / persona 实验前的 base 模型。

结构：

```text
token embedding
Transformer Block × N
  RMSNorm
  causal self-attention with RoPE on Q/K
  residual
  RMSNorm
  SwiGLU FFN
  residual
final RMSNorm
LM Head
```

已实现现代组件：

- RoPE：位置注入 Q/K。
- RMSNorm：替代 LayerNorm。
- SwiGLU：替代普通 GELU MLP。
- Pre-Norm：Norm 放在 attention / FFN 前。
- Weight Tying：可选，默认开启。
- 少 bias：可配置，默认关闭。

---

## 为什么不实现 MoE

MoE 是重要方向，但不适合本节。

原因：

1. MoE 会引入 router、expert、load balancing、aux loss、dispatch / combine 等新问题。
2. 它会明显增加工程复杂度，分散本节“训练工程底座”的目标。
3. 在 M1 Pro 小规模实验里，MoE 的收益不容易体现。
4. MoE 更适合作为后续单独专题。

当前只在报告中保留 MoE 方向，不在代码中实现。

---

## 暂不实现的现代组件

暂不实现：

- GQA / MQA。
- QK-Norm。
- Flash Attention。
- KV Cache 进阶优化。
- YaRN / LongRoPE 等长上下文扩展。
- MoE。
- chat template。
- LoRA。

原因：

这些都重要，但应该逐课引入。当前最重要的是把底座打稳。

---

## 本节 smoke test 结果

baseline_debug：

```text
run: 20260612_104206_baseline_debug
parameter_count: 23844
max_iters: 40
tokens_seen: 5120
final_train_loss: 1.9703
final_val_loss: 1.8672
tokens/sec: 13968.21
```

qwen_dense_tiny_debug：

```text
run: 20260612_104212_qwen_dense_tiny_debug
parameter_count: 18656
max_iters: 40
tokens_seen: 5120
final_train_loss: 1.8429
final_val_loss: 1.6612
tokens/sec: 13764.32
```

解释：

这不是严肃模型对比，因为数据很小、训练很短。这里只说明两类模型 profile 都能正常 forward、loss、backward、update、checkpoint、generate。
