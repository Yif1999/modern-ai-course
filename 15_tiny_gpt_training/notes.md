# 15：训练一个 Tiny GPT

## 这一课解决什么问题？

前面几课我们已经分别学过：

```text
token / vocab / embedding
position embedding
self-attention
multi-head attention
Transformer Block
```

这一课把这些组件组合起来，训练一个真正的小型自回归语言模型：

```text
Tiny GPT
```

它仍然很小，只使用字符级 tokenizer 和很短的训练文本，但结构上已经具备 GPT 的核心雏形。

---

## Tiny GPT 和 Bigram 的区别

Bigram 模型只看当前 token 来预测下一个 token：

```text
当前 token → 下一个 token
```

它几乎没有真正的上下文理解能力。

Tiny GPT 可以看一段上下文：

```text
前面多个 token → 下一个 token
```

原因是它有：

```text
position embedding
causal self-attention
multi-head attention
Transformer Blocks
```

所以 Tiny GPT 能根据更长的前文来判断下一个字符更可能是什么。

---

## Tiny GPT 和单层 Transformer Block 的区别

上一课只有一个 Transformer Block，重点是理解一个 block 内部：

```text
LayerNorm
Multi-Head Attention
Residual
FeedForward MLP
Residual
```

这一课把多个 block 堆叠起来：

```text
Transformer Block 1
↓
Transformer Block 2
↓
...
```

每个 block 都有自己的 attention、MLP、LayerNorm 参数。

堆叠之后，模型可以逐层加工 token 表示。浅层可能学习局部字符模式，后面的层可以在前一层表示的基础上组合更复杂的上下文关系。

---

## Tiny GPT 的完整数据流

完整流程是：

```text
文本
↓
字符级 tokenizer
↓
token ids: [batch, seq_len]
↓
token embedding: [batch, seq_len, n_embd]
↓
position embedding: [seq_len, n_embd]
↓
相加得到 x: [batch, seq_len, n_embd]
↓
Transformer Block × num_layers
↓
final LayerNorm: [batch, seq_len, n_embd]
↓
LM Head
↓
logits: [batch, seq_len, vocab_size]
↓
cross entropy
```

训练目标仍然是 next-token prediction。

---

## 为什么要堆叠多个 Transformer Blocks？

一个 block 已经可以让每个 token 读取前文信息。

但一个 block 的表达能力有限。

堆叠多个 block 后：

1. 每层都可以重新计算上下文关系。
2. 后一层可以基于前一层已经加工过的表示继续推理。
3. MLP 可以在每个 token 位置上加入更强的非线性变换。
4. 模型可以学习更复杂的字符组合和句子模式。

真实 GPT 也是这个思路，只是 block 数量、隐藏维度、训练数据都大得多。

---

## 关键超参数

`block_size` 表示上下文长度。

例如：

```text
block_size = 32
```

表示模型一次最多看 32 个 token。

`batch_size` 表示一次训练多少条序列。

例如：

```text
batch_size = 32
```

表示一次训练 32 条长度为 `block_size` 的序列。

`n_embd` 表示每个 token 的隐藏向量维度。

例如：

```text
n_embd = 64
```

表示每个 token 会被表示成 64 维向量。

`num_heads` 表示 multi-head attention 中有多少个 head。

例如：

```text
num_heads = 4
```

表示把 64 维表示拆成 4 个 head，每个 head 是 16 维。

`num_layers` 表示堆叠多少个 Transformer Blocks。

例如：

```text
num_layers = 2
```

表示模型有 2 层 Transformer Block。

---

## 为什么多个 block 后 shape 仍然不变？

每个 Transformer Block 的输入输出都设计成同一个 shape：

```text
[batch, seq_len, n_embd]
```

Attention 子层输出：

```text
[batch, seq_len, n_embd]
```

FeedForward MLP 虽然中间会扩展到：

```text
n_embd → 4 * n_embd
```

但最后又投影回：

```text
4 * n_embd → n_embd
```

所以 block 的输出仍然是：

```text
[batch, seq_len, n_embd]
```

这就是为什么 block 可以一层一层堆叠。

---

## logits 的 shape 为什么是 [batch, seq_len, vocab_size]？

语言模型要在每个位置预测下一个 token。

如果输入是：

```text
[batch, seq_len]
```

经过 embedding 和 Transformer 后，每个位置都有一个隐藏向量：

```text
[batch, seq_len, n_embd]
```

LM Head 会把每个位置的隐藏向量映射成词表大小的分数：

```text
n_embd → vocab_size
```

所以输出是：

```text
[batch, seq_len, vocab_size]
```

这表示：

1. batch 中每条序列都要预测。
2. 每条序列的每个位置都要预测。
3. 每个位置都对 vocab 中所有 token 给一个分数。

---

## cross entropy 在这里预测什么？

这里的 cross entropy 预测的是下一个 token。

训练数据中：

```text
x_batch = 当前 token 序列
y_batch = 向右移动一位的目标 token 序列
```

例如：

```text
x: h e l l
y: e l l o
```

含义是：

```text
看到 h，预测 e
看到 h e，预测 l
看到 h e l，预测 l
看到 h e l l，预测 o
```

训练时会把 `[batch, seq_len, vocab_size]` 的 logits 展平成 `[batch * seq_len, vocab_size]`，再和 `[batch * seq_len]` 的目标 token id 计算 cross entropy。

---

## generate 为什么只取最后一个位置的 logits？

训练时，一个序列里的每个位置都可以作为监督信号。

但生成时，我们当前只有一个上下文，要决定“下一个 token 是什么”。

例如上下文是：

```text
hello
```

模型会输出每个位置的 logits：

```text
h 的下一个
e 的下一个
l 的下一个
l 的下一个
o 的下一个
```

真正要继续生成的是：

```text
o 后面的下一个 token
```

所以生成时只取最后一个位置的 logits：

```text
logits[:, -1, :]
```

然后 softmax / sample 得到下一个 token，把它接到序列后面，再重复这个过程。

---

## 为什么现在生成文本仍然比较弱？

原因很直接：

1. 训练文本很小。
2. 使用的是字符级 tokenizer。
3. 模型规模很小。
4. 训练步数有限。
5. 没有使用真实大语料。
6. 没有复杂 tokenizer、dropout、学习率调度等训练技巧。

所以它能学到一些训练文本里的局部模式，但还不会生成自然、稳定、长期连贯的文本。

这不是结构错了，而是规模和数据还很小。

---

## 下一步可以如何改进？

可以从几个方向继续改进：

1. 增加训练文本。
2. 增加 `block_size`，让模型看到更长上下文。
3. 增加 `n_embd`，提高每个 token 的表示能力。
4. 增加 `num_layers`，让模型有更多 Transformer Blocks。
5. 增加 `num_heads`，让模型从更多关系角度读取上下文。
6. 使用更好的 tokenizer。
7. 加入保存和加载模型权重的完整流程。
8. 学习 KV Cache，提高生成速度。

---

## 这节课的核心结论

Tiny GPT 不是一个全新概念。

它就是把前面学过的组件按 GPT 的方式组合起来：

```text
token ids
↓
token embedding + position embedding
↓
Transformer Block × N
↓
final LayerNorm
↓
LM Head
↓
next-token prediction
```

从这一课开始，我们已经有了一个从零训练的最小 GPT 雏形。
