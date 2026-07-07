# 第 25 课：现代 LLM 架构升级实验

## 这一课解决什么问题？

前面我们训练的 Tiny GPT 使用的是比较朴素的 decoder-only Transformer：

```text
token embedding
+ learned position embedding
↓
Transformer Block × N
  LayerNorm
  Multi-Head Causal Self-Attention
  GELU FeedForward
↓
final LayerNorm
↓
LM Head
```

这已经是 GPT 的基本形状。

但现代 LLM 通常不会完全使用这个最朴素版本，而会加入一些更现代的组件，例如：

```text
RoPE
RMSNorm
SwiGLU
Weight Tying
GQA / MQA
```

本节目标不是扩大模型，而是理解这些结构替换后，shape 为什么仍然兼容，以及它们对 loss、速度、参数量有什么影响。

---

## 本节对比的两个模型

### baseline_tiny_gpt

使用旧版结构：

```text
learned position embedding
LayerNorm
GELU FeedForward
普通 Multi-Head Attention
独立 LM Head
```

### modern_tiny_gpt

使用现代化结构：

```text
RoPE
RMSNorm
SwiGLU FeedForward
Weight Tying
普通 Multi-Head Attention
```

注意：

```text
本节没有完整实现 GQA / MQA，只做概念说明。
```

---

## RMSNorm 是什么？

LayerNorm 的基本逻辑是：

```text
先减均值
再除以标准差
再乘可学习缩放参数
```

RMSNorm 更简单：

```text
不减均值
只按均方根 RMS 缩放
再乘可学习缩放参数
```

公式直觉：

```text
RMS(x) = sqrt(mean(x²))
output = x / RMS(x) * weight
```

RMSNorm 常用于现代 LLM，因为它更简单，计算更少，也能提供足够好的归一化效果。

---

## RoPE 是什么？

旧版 position embedding 是：

```text
x = token_embedding + position_embedding
```

也就是说，位置信息被直接加到 token 表示上。

RoPE 不这样做。

RoPE 的做法是：

```text
对 attention 里的 Q 和 K 注入位置信息
```

具体来说，RoPE 会把 Q/K 的成对维度当成二维平面，然后根据 token 位置做旋转。

直觉上：

```text
同一个 token 在不同位置，Q/K 的方向会不同
```

这样 attention score：

```text
Q @ Kᵀ
```

就能感知相对位置关系。

RoPE 常用于现代 decoder-only LLM，例如 LLaMA 系列。

---

## 为什么 RoPE 作用在 Q/K？

attention 的核心是：

```text
scores = Q @ Kᵀ
```

Q 表示当前位置要找什么信息。

K 表示每个位置提供什么匹配线索。

所以把位置信息注入 Q/K，等于让“匹配关系”本身带上位置信息。

V 主要是被加权汇总的内容本身，通常不需要 RoPE。

---

## SwiGLU 是什么？

普通 FFN 是：

```text
Linear
GELU
Linear
```

SwiGLU 是门控结构：

```text
gate = Linear(x)
value = Linear(x)
hidden = SiLU(gate) * value
out = Linear(hidden)
```

它多了一条 gate 分支。

直觉上：

```text
value 提供内容
gate 决定哪些内容通过
```

这种门控机制比普通 GELU MLP 表达能力更强，所以现代 LLM 中很常见。

代价是：

```text
参数和计算通常会增加
```

---

## Weight Tying 是什么？

语言模型有两个和 vocab_size 相关的大矩阵：

```text
输入 embedding table: [vocab_size, n_embd]
输出 LM Head:         [n_embd, vocab_size]
```

如果不做 weight tying，这两个矩阵是独立参数。

Weight Tying 的做法是：

```text
logits = hidden @ token_embedding_table.T
```

也就是输出层直接复用输入 embedding table。

好处：

```text
减少参数
让输入 token 表示和输出 token 分类空间共享
```

在 vocab_size 较大时，省下的参数很明显。

---

## GQA / MQA 是什么？

本节没有完整实现 GQA / MQA，但需要理解概念。

### MHA

Multi-Head Attention：

```text
每个 attention head 都有自己的 Q/K/V
```

优点是表达能力强。

缺点是推理时 KV Cache 占用较大。

### MQA

Multi-Query Attention：

```text
多个 Q heads 共享同一组 K/V
```

优点是显著减少 KV Cache。

缺点是可能降低表达能力。

### GQA

Grouped-Query Attention：

```text
多组 Q heads 共享较少组 K/V
```

它介于 MHA 和 MQA 之间。

现代 LLM 常用 GQA，因为它在表达能力和推理效率之间更平衡。

---

## Shape 为什么保持兼容？

虽然组件变了，但每个 Transformer Block 的输入输出 shape 仍然保持：

```text
[batch, seq_len, n_embd]
```

原因是：

```text
RoPE 只旋转 Q/K，不改变 shape
RMSNorm 输入输出 shape 一样
SwiGLU 最后投影回 n_embd
Weight Tying 只改变 LM Head 的参数来源，不改变 logits shape
```

最终 logits 仍然是：

```text
[batch, seq_len, vocab_size]
```

---

## 本节重点看什么？

本节不是要证明 modern 一定立刻更好。

小模型、短训练、少数据下，现代组件的优势可能不明显。

重点是看：

```text
结构是否能正常训练
shape 是否兼容
参数量是否变化
tokens/sec 是否变化
loss 是否变化
生成文本是否有变化
```

---

## 当前实验和真实现代 LLM 的差距

真实现代 LLM 通常还有：

```text
更大模型
更大数据
更长训练
学习率 warmup
cosine decay
dropout / weight decay 策略
更大的 batch
更好的 tokenizer
GQA
KV Cache 优化
混合精度与分布式训练
```

所以本节只是结构理解实验，不代表完整现代 LLM recipe。

---

## 本次实际实验结果

本次使用第 24 课已经处理好的 medium 中文 BPE token 数据。

训练配置：

```text
block_size = 128
batch_size = 16
n_embd = 64
num_heads = 4
num_layers = 2
max_iters = 1000
learning_rate = 2e-3
tokens_seen = 2,048,000
```

结果：

| run | 参数量 | train loss | val loss | tokens/sec |
| --- | ---: | ---: | ---: | ---: |
| baseline_tiny_gpt | 1,164,672 | 6.6257 | 6.7615 | 125,147.6 |
| modern_tiny_gpt | 665,152 | 6.7215 | 6.8444 | 107,755.1 |

### 结果解释

在这次短训练里：

```text
baseline 的 val loss 更低
modern 的参数量更少
modern 的速度略慢
```

modern 参数量从 `1,164,672` 降到 `665,152`，主要来自 Weight Tying 省掉了独立 LM Head 的大矩阵。

但 modern 没有在 loss 上赢过 baseline。

这说明：

```text
现代组件不是一换就必然更好
```

可能原因：

1. 训练步数太短。
2. 模型太小。
3. learning_rate 没有为 modern 单独调。
4. SwiGLU 和 RoPE 有额外计算，小模型下不一定更快。
5. Weight Tying 减少参数后，短训练下表达能力可能略受影响。

### 哪些组件最值得保留？

从本次实验看：

```text
Weight Tying 最值得保留
```

原因是它的参数减少最直观，而且不改变 logits shape。

RMSNorm 和 RoPE 更符合现代 LLM 结构习惯，也值得保留，但它们的优势需要更长训练和更合适的 recipe 才更明显。

SwiGLU 提升表达能力，但会增加 FFN 计算，小模型短训练下不一定立刻体现收益。

### 本节结论

这节课的重点不是证明 modern 一定赢。

更重要的结论是：

```text
现代 LLM 架构升级可以保持输入输出 shape 不变
但会改变参数分布、训练动态、速度和归纳偏置
```

本次实验说明：

```text
在当前小模型和短训练预算下，baseline loss 略好；
modern 参数效率更高，但需要更完整的训练 recipe 才可能体现优势。
```

因此可以进入下一课：

```text
26_open_training_recipe_review
```

下一步应该把架构组件、学习率、warmup、训练步数、数据规模和采样策略放到一个完整 training recipe 里看，而不是只比较单个结构替换。
