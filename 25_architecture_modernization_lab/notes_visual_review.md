# 25b：现代 LLM 架构组件动画复盘

## 新版动画看什么？

这版动画不再把组件画成抽象流程图，而是尽量按“矩阵 / 向量实际怎么计算”来拆。

生成了一个总视频和四个独立片段：

```text
outputs/animations/modern_architecture_components.mp4
outputs/animations/detail_rope_qk_matrix.mp4
outputs/animations/detail_rmsnorm.mp4
outputs/animations/detail_swiglu.mp4
outputs/animations/detail_weight_tying.mp4
```

重点观察：

1. RoPE：先看 Q/K 矩阵每一行按 position 旋转，再看单行向量如何拆成二维 pair 旋转，最后看 `Q_rot @ K_rot.T` 生成 attention scores。
2. RMSNorm：看 hidden vector 如何平方、求 mean、求 RMS，再除以 RMS。
3. SwiGLU：看 `x` 分别乘 `W_gate` 和 `W_value`，得到两条向量，再做 `SiLU(gate) * value`。
4. Weight Tying：看同一张 embedding table 如何既用于输入查表，也用于输出 `hidden @ E.T` 得到 logits。

## RoPE

### 它替代了什么？

RoPE 替代了传统的 learned position embedding。

传统写法是：

```text
x = token_embedding + position_embedding
```

### 它做了什么？

RoPE 不把 position embedding 直接加到 `x` 上。

它是在 attention 里面，把位置信息注入到：

```text
Q
K
```

具体方式是对 Q/K 的成对维度做旋转。

不同位置对应不同旋转角度。

### 它为什么有用？

attention score 来自：

```text
Q @ K.T
```

如果 Q/K 本身带有位置信息，那么点积结果就能感知相对位置。

所以 RoPE 很适合 decoder-only LLM 的自回归 attention。

### 代码里通常出现在哪里？

通常出现在 attention 模块里：

```text
x -> Q/K/V projection
Q, K -> apply_rope
scores = Q_rot @ K_rot.T
```

V 通常不做 RoPE。

---

## RMSNorm

### 它替代了什么？

RMSNorm 替代了 LayerNorm。

LayerNorm 会：

```text
减均值
除标准差
```

### 它做了什么？

RMSNorm 主要做：

```text
RMS = sqrt(mean(x_i^2))
x_norm = x / RMS
out = scale * x_norm
```

它不减均值。

### 它为什么有用？

RMSNorm 更简单，计算更少。

它的核心作用是控制 hidden state 的尺度，让每层输入更稳定。

现代 LLM 经常使用 RMSNorm。

### 代码里通常出现在哪里？

通常出现在 Transformer Block 里：

```text
x = x + attention(rmsnorm(x))
x = x + ffn(rmsnorm(x))
```

也就是 attention 和 FFN 前面。

---

## SwiGLU

### 它替代了什么？

SwiGLU 替代普通 GELU FeedForward。

普通 FFN 是：

```text
Linear -> GELU -> Linear
```

### 它做了什么？

SwiGLU 是双分支门控结构：

```text
gate = Linear(x)
value = Linear(x)
hidden = SiLU(gate) * value
out = Linear(hidden)
```

### 它为什么有用？

gate 分支决定哪些信息通过。

value 分支提供被调制的信息。

这种门控结构通常比普通 GELU MLP 表达能力更强。

### 代码里通常出现在哪里？

通常替代 Transformer Block 里的 FFN：

```text
self.ffn = SwiGLUFeedForward(...)
```

然后：

```text
x = x + self.ffn(norm(x))
```

---

## Weight Tying

### 它替代了什么？

它替代独立的 LM Head 权重。

不做 weight tying 时有两张大矩阵：

```text
token_embedding_table: [vocab_size, n_embd]
lm_head_weight:        [n_embd, vocab_size]
```

### 它做了什么？

Weight Tying 让输出层复用输入 embedding table：

```text
logits = hidden @ token_embedding_table.T
```

### 它为什么有用？

它减少参数。

它还让输入 token 表示和输出 token 分类空间形成对应关系。

在 vocab_size 很大时，省下的参数非常明显。

### 代码里通常出现在哪里？

通常出现在模型最后：

```text
x = final_norm(x)
logits = x @ token_embedding_table.T
```

也就是 LM Head 的位置。

---

## 总结

现代 LLM 架构升级可以理解成：

```text
RoPE：负责位置
RMSNorm：稳定尺度
SwiGLU：增强前馈表达
Weight Tying：减少参数并共享词表表示
```

这些组件通常不会改变主干 shape：

```text
[batch, seq_len, n_embd]
```

但它们会改变：

```text
位置信息注入方式
归一化方式
前馈网络表达能力
参数共享方式
训练动态
```

所以现代 LLM 不只是“更大的 Transformer”，也包含很多结构和工程上的改进。
