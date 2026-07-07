# 17：KV Cache 推理优化入门

## 这一课解决什么问题？

前面 Tiny GPT 已经可以按自回归方式生成文本：

```text
给定 prompt
↓
模型预测下一个 token
↓
把新 token 接回输入
↓
继续预测下一个 token
```

普通生成可以工作，但有一个明显浪费：

每生成一个新 token，模型都会重新计算整段上下文。

KV Cache 的目标就是减少这种重复计算。

---

## 普通自回归生成流程

假设 prompt 是：

```text
hello
```

普通生成第一个新 token 时，模型计算：

```text
h e l l o
```

生成一个新 token 后，下一步又计算：

```text
h e l l o 新token
```

再下一步又计算：

```text
h e l l o 新token 新token
```

问题是：

前面的 `h e l l o` 每一步都被重新送进 Transformer。

---

## 重复计算在哪里？

在 self-attention 里，每个 token 都会产生三组向量：

```text
Query
Key
Value
```

普通生成时，每一步都会重新计算所有历史 token 的 Q/K/V。

但历史 token 的 Key 和 Value 在推理时不会因为未来 token 改变。

所以它们可以缓存下来。

---

## KV Cache 的核心思想

KV Cache 保存的是过去 token 在每一层 attention 里产生的：

```text
Key
Value
```

生成新 token 时，只需要：

1. 为新 token 计算新的 Q/K/V。
2. 把新 token 的 K/V 追加到 cache。
3. 用新 token 的 Q 去和 cache 里的所有 K 做 attention。
4. 用 attention weights 对 cache 里的所有 V 加权求和。

这样就不用每一步重新计算历史 token 的 K/V。

---

## Query、Key、Value 各自的角色

可以粗略理解成：

```text
Query：当前 token 想找什么信息
Key：每个 token 提供的可匹配索引
Value：每个 token 真正提供的内容
```

attention 的核心计算是：

```text
scores = Q @ K.T / sqrt(head_size)
weights = softmax(scores)
output = weights @ V
```

新 token 的 Query 会和所有历史 Key 匹配，然后从对应 Value 里读取信息。

---

## 为什么缓存 K/V，而不是缓存 Q？

生成第 t 个 token 时，我们只需要当前 token 的 Query。

过去 token 的 Query 主要用于“过去 token 自己作为当前位置时”读上下文。

但生成当前 token 时，真正需要的是：

```text
当前 Query
过去和当前的 Key
过去和当前的 Value
```

所以 cache 保存 K/V 就够了。

Query 每一步都只为当前 token 现算，不需要长期保存。

---

## 单层 KV Cache 的 shape

本课模型配置里：

```text
num_heads = 4
head_size = 16
```

单层 cache 的常见 shape 是：

```text
key:   [batch, num_heads, cache_seq_len, head_size]
value: [batch, num_heads, cache_seq_len, head_size]
```

例如 batch=1、生成到第 6 个 token：

```text
key shape   = [1, 4, 6, 16]
value shape = [1, 4, 6, 16]
```

其中 `cache_seq_len` 会随着生成增长。

---

## 多层 Transformer 为什么每层都要有 cache？

Tiny GPT 有多个 Transformer Block。

每一层的输入隐藏状态都不同，所以每一层 attention 产生的 Key/Value 也不同。

因此不能所有层共用一份 cache。

多层模型需要：

```text
layer 0 cache
layer 1 cache
layer 2 cache
...
```

每层都有自己的 K/V。

---

## KV Cache 对内存的影响

KV Cache 用计算换内存。

它减少了重复计算，但需要保存历史 token 的 K/V。

上下文越长，cache 越大：

```text
cache 大小 roughly ∝ num_layers × num_heads × seq_len × head_size
```

在 Apple Silicon 上，这些 cache 会占用统一内存。

真实大模型长上下文推理时，KV Cache 往往是主要内存开销之一。

---

## 训练阶段和推理阶段的区别

训练阶段通常不用 KV Cache。

原因是训练时要一次性计算整段序列所有位置的 loss：

```text
logits: [batch, seq_len, vocab_size]
targets: [batch, seq_len]
```

这可以并行完成。

而推理阶段是一个 token 一个 token 顺序生成：

```text
生成 token 1
生成 token 2
生成 token 3
...
```

所以推理阶段更适合使用 KV Cache。

---

## KV Cache 的局限

KV Cache 不是免费优化。

它有几个限制：

1. cache 会随着上下文长度增长。
2. 长上下文会占用更多内存。
3. batch 推理时，每条样本长度可能不同，cache 管理更复杂。
4. 如果超过模型的 context window，仍然需要截断、滑动窗口或其他长上下文策略。
5. 教学版 Python 实现不一定在小模型上更快，因为框架调度和循环开销很明显。

---

## 这节课的核心结论

KV Cache 不改变模型输出的数学含义。

它改变的是推理时的计算方式：

```text
无 cache：每一步重新计算整个上下文
有 cache：只计算新 token，并复用历史 K/V
```

如果实现正确，在 greedy decoding 下：

```text
no-cache 生成结果
with-cache 生成结果
```

应该一致或只有极小数值误差。

真实大模型中 KV Cache 非常重要，因为生成长文本时，它可以显著减少重复计算。

---

## 下一步

从教学版 KV Cache 继续往真实推理系统走，可以学习：

1. 更高效的 cache 数据结构。
2. batch generation 的 cache 管理。
3. prompt prefill 和 decode 阶段分离。
4. 长上下文下的 sliding window / paged attention。
5. KV Cache 和量化、并发推理之间的关系。

本课先不进入这些优化，只理解 KV Cache 的基本计算结构。
