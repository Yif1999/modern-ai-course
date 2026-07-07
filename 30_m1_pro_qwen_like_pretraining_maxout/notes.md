# 第 30 课：M1 Pro 上的 Qwen-like 预训练极限实验

## 2026-06-13 更新：主 tokenizer 改为自训 lab tokenizer

本课最初尝试使用 `Qwen/Qwen3.6-27B` tokenizer。

probe 后发现：

- Qwen vocab size 为 `248,077`。
- 当前 1000 万字符数据实际只用到一部分 token。
- 大词表显著增加 embedding / LM Head / logits 成本。
- 使用 Qwen tokenizer 并不会让我们的自训模型兼容官方 Qwen 权重。

因此本课主线调整为：

```text
自训 lab ByteLevel BPE tokenizer
vocab_size = 32768
context = 1024
Qwen-like dense decoder-only architecture
M1 Pro maxout probe
```

Qwen tokenizer 仍保留为未来真实 Qwen / MLX-LM / LoRA / SFT 的长度统计工具，但不作为当前自训基座模型的主 tokenizer。

最新 maxout 结论：

- 0.35B：最均衡，约 652 tokens/sec，MLX peak 约 8.30GB。
- 0.55B：更激进，约 325 tokens/sec，MLX peak 约 12.22GB。
- 0.75B：能跑但明显慢，约 174 tokens/sec。
- 1.0B：能跑但不适合当前长训，约 22.7 tokens/sec。

第一晚建议跑：

```text
configs/overnight_lab_balanced_0p35b.json
```

如果想更激进，再跑：

```text
configs/overnight_lab_aggressive_0p55b.json
```

## 这一课为什么改变方向？

原计划第 30 课进入 persona SFT。

但我们当前的基座模型还很弱：

- 训练 token 不够多。
- 自训 lab tokenizer 覆盖不足。
- 小模型训练时间过短。
- 前面很多实验更像 smoke test，而不是认真训练。

如果现在直接做 SFT，很难判断生成效果差是因为：

```text
基座太弱
数据不好
SFT 写法不对
tokenizer 不合适
训练时间不够
```

所以第 30 课先回到基座训练，把它当成一次更严谨的本地训练工程实验。

## 本课目标

在 M1 Pro / 32GB 统一内存上，用足当前硬件能力，训练一个更接近真实 decoder-only LLM 的中文小模型。

本课重点不是马上得到好模型，而是建立一套更像企业内部训练实验的流程：

1. 明确数据来源。
2. 使用成熟 tokenizer。
3. 使用现代 dense decoder-only 架构。
4. 使用 1K context。
5. 做吞吐、内存、loss probe。
6. 接入 Dashboard。
7. 为夜间长训练选择配置。

## Tokenizer

本课使用：

```text
Qwen/Qwen3.6-27B tokenizer
```

实际 vocab size：

```text
248,077
```

优点：

- 中文覆盖更好。
- 中英文混排、数字、标点、特殊符号更成熟。
- 未来可以衔接真实 Qwen / MLX-LM / LoRA / SFT。

代价：

- 词表很大。
- logits 最后一维很大。
- 小模型里 embedding / LM Head 参数占比会很高。

## 数据

本课目前复用前面已经下载和清洗的数据：

- 第 23 课真实中文开源数据
- 第 24 课 scaling 数据
- 第 28 课中文趣味语料

当前 corpus：

```text
10,000,004 字符
6,136,780 Qwen tokens
```

train / val：

```text
train tokens: 6,014,044
val tokens: 122,736
```

## 模型结构

本课使用 Qwen dense-like 模型。

包含：

- RoPE
- RMSNorm
- SwiGLU
- GQA
- QK-Norm
- Weight Tying
- bfloat16
- no bias
- causal self-attention

暂不包含：

- Qwen3.6 的 Gated DeltaNet
- 线性注意力混合架构
- MTP
- MoE
- 超长上下文扩展

原因：

本课 context 只有 1024，线性注意力的主要价值在超长上下文。当前阶段更重要的是把 dense decoder-only 训练做到稳定、可观察、可复现。

## 为什么 context 是 1024？

1024 token 对当前实验是一个合理上限：

- 比前面 64 / 128 更接近真实语言模型训练。
- 可以覆盖较长中文段落和对话。
- 注意力计算仍能在 M1 Pro 上承受。
- 足够观察 RoPE、GQA、SwiGLU 等结构的真实训练表现。

## Probe 结果

当前已经跑通：

```text
probe_micro_256
probe_120m_context1024
probe_180m_context1024
```

关键观察：

- 120M / 1K context：约 1.18 亿参数，约 602 tokens/sec，MLX peak 约 5.24GB。
- 180M / 1K context：约 1.77 亿参数，约 502 tokens/sec，MLX peak 约 6.46GB。

两者都能跑。

120M 更划算，180M 更激进。

## Dashboard 要看什么？

长训练时重点看：

1. train loss 是否下降。
2. val loss 是否下降。
3. tokens/sec 是否稳定。
4. step time 是否异常变慢。
5. MLX peak memory 是否接近机器上限。
6. samples 是否从乱码逐步变成中文片段。
7. checkpoints 是否正常保存。

不要只看 GPU 占用率。

Apple Silicon 上没有稳定的普通用户态 MLX API 直接给 GPU utilization。我们先用 tokens/sec、step time、MLX/Metal memory 作为训练性能信号。

## 当前推荐

早期建议曾经是：

```text
overnight_candidate_120m
```

但这属于 Qwen tokenizer 阶段的旧配置。现在主线已经切换到：

```text
LCCC-first canonical corpus + 自训练 ByteLevel BPE tokenizer + 0.35B / 0.55B Qwen-like dense 模型
```

原因：

- 自训练 tokenizer 的 vocab_size 是 32768，比 Qwen 大词表更适合当前自训小模型。
- 新 canonical 数据以 LCCC 中文社媒短对话为首位，占比约 65.7%。
- SkyPile / ChineseWebText / 少量 FineWeb 只做网页补充。
- 政府/报告、新闻、法律、医疗、招聘等文体被压低，避免模型输出过度公文风。
- 0.35B 在 M1 Pro 上可跑，peak 约 8.3GB，速度约 600 tokens/sec。
- 0.55B 可作为第二阶段激进配置，但第一晚更建议先跑 0.35B。

当前推荐第一晚训练：

```text
overnight_lab_balanced_0p35b
```

第二晚或周末可以尝试：

```text
overnight_lab_aggressive_0p55b
```

用来验证更大容量是否在更长训练后兑现。

## 本课核心结论

第 30 课不是玩具 SFT。

它是一次本地生产化预训练实验的起点：

```text
更真实的 tokenizer
更长 context
更现代的架构
更长训练时间
更完整 telemetry
更严肃的实验记录
```

只有基座训练更清楚之后，再进入 persona SFT / LoRA / MLX-LM 才更有意义。
