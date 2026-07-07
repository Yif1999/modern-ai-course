# 13：Multi-Head Attention 多头注意力

## 这一课解决什么问题？

上一课我们学习了 single-head causal self-attention。

单头注意力已经能让每个 token 读取前面的 token。

但是只有一个 head，就像只有一个观察角度。

这一课我们引入：

```text
Multi-Head Attention
```

也就是多头注意力。

它的核心思想是：

```text
多个 attention head 并行工作
每个 head 可以学习不同的关注关系
最后把多个 head 的结果合并起来
```

---

## 从 single-head 到 multi-head

single-head attention 的流程是：

```text
x
↓
Q, K, V
↓
QKᵀ / sqrt(d)
↓
causal mask
↓
softmax
↓
attention weights @ V
↓
out
```

multi-head attention 的流程是：

```text
x
↓
Q, K, V
↓
切分成多个 heads
↓
每个 head 独立做 causal self-attention
↓
concat 所有 heads 的输出
↓
output projection
```

---

## 为什么需要多个 head？

不同 head 可以关注不同类型的关系。

例如在语言里，一个 head 可能更关注：

```text
相邻字符
```

另一个 head 可能更关注：

```text
空格或单词边界
```

另一个 head 可能更关注：

```text
重复出现的模式
```

在真实 GPT 里，不同 head 可能学习到更复杂的关系，比如语法、引用、长距离依赖等。

可以简单理解为：

```text
single-head：一个观察视角
multi-head：多个观察视角
```

---

## 关键 shape

本节设置：

```text
n_embd = 32
num_heads = 4
head_size = 8
```

因为：

```text
head_size = n_embd // num_heads = 32 // 4 = 8
```

输入隐藏向量是：

```text
x: [batch, seq_len, n_embd]
```

投影后：

```text
q/k/v: [batch, seq_len, n_embd]
```

切分 head 后：

```text
q_heads/k_heads/v_heads: [batch, num_heads, seq_len, head_size]
```

attention scores 是：

```text
scores: [batch, num_heads, seq_len, seq_len]
```

每个 head 都有自己的 attention map。

---

## 为什么 scores 多了一个 head 维度？

单头 attention 的 scores 是：

```text
[batch, seq_len, seq_len]
```

多头 attention 的 scores 是：

```text
[batch, num_heads, seq_len, seq_len]
```

也就是说：

```text
每个 batch 样本
每个 head
每个当前位置
对序列中每个位置
都有一个注意力分数
```

所以每个 head 都有自己的注意力矩阵。

---

## concat 是什么？

每个 head 输出：

```text
[batch, seq_len, head_size]
```

有 num_heads 个 head。

把它们拼起来后：

```text
[batch, seq_len, num_heads * head_size]
```

因为：

```text
num_heads * head_size = n_embd
```

所以 concat 后得到：

```text
[batch, seq_len, n_embd]
```

---

## 为什么还要 output projection？

多个 head concat 后，只是把不同 head 的输出拼在一起。

output projection 的作用是：

```text
重新混合不同 head 的信息
```

代码里是：

```python
self.proj(out)
```

它会把拼接后的多头信息重新投影回模型的隐藏空间。

---

## Multi-Head Attention 和 Single-Head Attention 的区别

Single-head：

```text
一个 Q/K/V 注意力视角
```

Multi-head：

```text
多个 Q/K/V 注意力视角并行
```

更直观地说：

```text
single-head 是一个人看上下文
multi-head 是多个人从不同角度看上下文
然后把他们的观察结果汇总
```

---

## 为什么还不是完整 Transformer Block？

本节只实现了：

```text
multi-head causal self-attention
```

完整 Transformer block 还需要：

```text
residual connection
LayerNorm
FeedForward MLP
Dropout
多个 block 堆叠
```

所以本节只是 GPT 核心组件之一。

下一步我们会把这些组件组合成真正的 Transformer Block。

---

## 这节课的核心结论

Multi-Head Attention 的核心是：

```text
把 n_embd 切成多个 head
每个 head 独立做 causal self-attention
再 concat 回 n_embd
最后用 output projection 混合信息
```

它让模型拥有多个并行的上下文观察角度。

这就是 GPT 中 attention 层的基本形态。
