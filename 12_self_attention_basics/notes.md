# 12：Self-Attention 基础

## 这一课解决什么问题？

上一课我们加入了：

```text
token embedding + position embedding
```

这让模型知道：

```text
这个 token 是谁
这个 token 在哪里
```

但它仍然缺少一个关键能力：

```text
当前 token 如何读取前面的 token？
```

Self-attention 就是为了解决这个问题。

---

## Self-Attention 的核心直觉

Self-attention 可以理解成：

```text
每个 token 都在问：
我应该看前面的哪些 token？
每个 token 应该看多少？
```

例如在一句话中：

```text
hello ai
```

当模型预测最后一个 token 后面应该接什么时，它可能需要关注前面的：

```text
h
e
l
l
o
空格
a
i
```

不同位置的重要性不同。

attention weights 就是在表达这种“关注程度”。

---

## Q、K、V 是什么？

Self-attention 会把每个 token 的隐藏向量变成三种向量：

```text
Q: Query
K: Key
V: Value
```

可以用一个类比理解：

```text
Query：我想找什么信息？
Key：我这里有什么信息？
Value：如果你关注我，我能提供什么内容？
```

计算过程是：

```text
Q 和 K 算相似度
相似度经过 softmax 变成 attention weights
attention weights 对 V 做加权求和
```

---

## Attention Scores

attention scores 的计算是：

```text
scores = QK^T / sqrt(head_size)
```

如果输入长度是 `seq_len`，那么 scores 的 shape 是：

```text
[batch, seq_len, seq_len]
```

含义是：

```text
对于每个样本
每个当前位置
它对序列中每个位置的关注分数
```

例如第 5 行表示：

```text
第 5 个 token 正在看哪些 token
```

---

## 为什么要除以 sqrt(head_size)？

如果 `head_size` 很大，Q 和 K 点积的数值可能会变得很大。

这会让 softmax 变得过于尖锐，训练不稳定。

所以要除以：

```text
sqrt(head_size)
```

来缩放分数。

这叫 scaled dot-product attention。

---

## 什么是 causal mask？

语言模型是预测下一个 token。

在训练时，位置 t 不能偷看未来位置 t+1、t+2、t+3。

所以我们需要 causal mask。

它是一个下三角矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

含义是：

```text
第 0 个位置只能看自己
第 1 个位置可以看 0 和 1
第 2 个位置可以看 0、1、2
第 3 个位置可以看 0、1、2、3
```

右上角的位置代表未来 token，必须屏蔽。

---

## Attention Weights 是什么？

经过 mask 和 softmax 后，得到：

```text
attention weights
```

shape 是：

```text
[batch, seq_len, seq_len]
```

每一行加起来等于 1。

它表示：

```text
当前位置把注意力分配给前面各个位置的比例。
```

可以简单理解成：

```text
我现在要预测下一个 token，
我应该从前面哪些 token 里取信息？
```

---

## out = attention_weights @ V 是什么？

attention weights 表示“看谁、看多少”。

V 表示“被看的 token 能提供什么信息”。

所以：

```text
out = attention_weights @ V
```

就是：

```text
根据注意力权重，把前面 token 的信息加权混合起来。
```

输出 shape 是：

```text
[batch, seq_len, head_size]
```

这个输出已经包含了上下文信息。

---

## 这节模型比上一课强在哪里？

上一课模型是：

```text
token embedding + position embedding
↓
linear head
```

它知道 token 和位置，但 token 之间没有真正交流。

这一课模型是：

```text
token embedding + position embedding
↓
Q, K, V
↓
causal self-attention
↓
linear head
```

它让每个位置可以读取自己和前面位置的信息。

所以它开始具备真正的上下文建模能力。

---

## 为什么还不是完整 GPT？

因为完整 GPT 还需要：

```text
multi-head attention
residual connection
LayerNorm
feed-forward MLP
多个 Transformer blocks 堆叠
```

本节只有：

```text
single-head causal self-attention
```

所以它只是 GPT 的核心零件之一。

---

## 这节课的核心结论

Self-attention 的核心公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

在语言模型里，还必须加上：

```text
causal mask
```

防止模型偷看未来。

Self-attention 让每个 token 可以根据注意力权重读取前面的 token。

这就是 GPT 能利用上下文的基础。
