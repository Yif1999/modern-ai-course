# 14：Transformer Block 基础

## 这一课解决什么问题？

上一课我们实现了 Multi-Head Attention。

Multi-Head Attention 已经能让 token 从多个角度读取上下文。

但是完整的 Transformer Block 不只有 attention。

它还需要：

```text
Residual Connection
LayerNorm
FeedForward MLP
```

这一课就是把这些组件组合成一个完整的 Transformer Block。

---

## Transformer Block 的基本结构

本节采用 Pre-LN 结构，流程是：

```text
输入 x
↓
LayerNorm
↓
Multi-Head Causal Self-Attention
↓
Residual Add
↓
LayerNorm
↓
FeedForward MLP
↓
Residual Add
↓
输出
```

更像代码里的：

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

---

## 为什么需要 residual connection？

Residual connection 中文常叫残差连接。

它的形式是：

```text
x = x + 子层输出
```

例如：

```text
x = x + attention(layer_norm(x))
```

它的作用是：

1. 保留原始信息。
2. 让梯度更容易传回前面的层。
3. 让模型可以学习“在原表示基础上做修改”，而不是每层都完全重写表示。

可以简单理解为：

```text
主路径做变换，残差路径保底传原信息。
```

---

## 为什么需要 LayerNorm？

LayerNorm 是层归一化。

它会对每个 token 的隐藏向量进行归一化。

如果隐藏向量 shape 是：

```text
[batch, seq_len, n_embd]
```

那么：

```text
LayerNorm(n_embd)
```

主要是在最后一维 `n_embd` 上做归一化。

作用是：

1. 稳定训练。
2. 避免不同层之间数值分布变化太大。
3. 让深层网络更容易优化。

---

## 为什么这里使用 Pre-LN？

Pre-LN 指的是：

```text
先 LayerNorm，再进入 attention 或 MLP
```

也就是：

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

另一种常见形式是 Post-LN：

```text
x = layer_norm(x + attention(x))
```

本节使用 Pre-LN，是因为它在深层 Transformer 中通常更稳定，也更符合很多现代 GPT 风格实现。

---

## FeedForward MLP 是什么？

Transformer Block 里的 MLP 通常是：

```text
n_embd → 4 * n_embd → n_embd
```

例如本节：

```text
32 → 128 → 32
```

它对每个 token 位置独立作用。

也就是说，它不会在不同 token 之间交流信息。

token 之间的信息交流主要由 attention 完成。

MLP 的作用是：

```text
增强每个位置自己的表示能力。
```

---

## 为什么 FeedForward 要先扩大再缩回？

常见结构是：

```text
n_embd
↓
4 * n_embd
↓
激活函数
↓
n_embd
```

扩大维度可以给模型更强的非线性表达空间。

缩回 `n_embd` 是为了保持 Block 输入输出 shape 一致，方便堆叠更多 Block。

---

## Transformer Block 的 shape

输入：

```text
[batch, seq_len, n_embd]
```

经过 attention 后：

```text
[batch, seq_len, n_embd]
```

经过 MLP 后：

```text
[batch, seq_len, n_embd]
```

输出仍然是：

```text
[batch, seq_len, n_embd]
```

这很重要，因为只有输入输出 shape 一致，才能把多个 Transformer Block 堆叠起来。

---

## 本节模型和上一课的区别

上一课只有：

```text
token embedding + position embedding
↓
multi-head attention
↓
lm_head
```

这一课变成：

```text
token embedding + position embedding
↓
Transformer Block
↓
final LayerNorm
↓
lm_head
```

Transformer Block 内部包含：

```text
Multi-Head Attention
Residual Connection
LayerNorm
FeedForward MLP
```

所以这一课更接近真正的 GPT。

---

## 为什么还不是完整 Tiny GPT？

因为本节只有一个 Transformer Block。

完整 Tiny GPT 通常会：

```text
堆叠多个 Transformer Blocks
加入更完整的模型封装
加入采样参数
保存模型
可能加入 dropout
```

但核心结构已经基本出现了。

下一步就是把多个 Transformer Block 堆起来，形成真正的 Tiny GPT。

---

## 这节课的核心结论

Transformer Block 可以理解为：

```text
Attention 负责读取上下文
Residual 负责保留信息和帮助梯度传播
LayerNorm 负责稳定训练
MLP 负责增强每个 token 的表达能力
```

它的输入输出 shape 保持一致：

```text
[batch, seq_len, n_embd]
```

所以可以一层一层堆叠。

这就是 GPT 的核心积木。
