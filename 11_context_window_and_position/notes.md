# 11：上下文窗口与位置 Embedding

## 这一课解决什么问题？

上一课我们训练了 Bigram 语言模型。

Bigram 模型只根据当前 token 预测下一个 token。

它的问题是：

```text
它看不到更长上下文
```

这一课我们引入两个 GPT 里非常重要的概念：

```text
context window：上下文窗口
position embedding：位置向量
```

不过本节还不引入 self-attention。

所以这节模型只是为后面的 Transformer 做准备。

---

## 什么是 context window？

context window 是模型一次能处理的 token 序列长度。

在代码里：

```python
block_size = 16
```

表示每次输入给模型的是 16 个 token。

输入 shape 是：

```text
[batch, seq_len]
```

例如：

```text
[16, 16]
```

含义是：

```text
一次输入 16 条序列
每条序列长度是 16 个 token
```

在真正的 GPT 里，context window 就是模型最多能看多长的上下文。

例如：

```text
2048 tokens
8192 tokens
128k tokens
```

都可以理解为不同大小的 context window。

---

## 为什么需要 position embedding？

文本是一串有顺序的 token。

例如：

```text
abc
```

和：

```text
cba
```

包含的字符一样，但顺序完全不同。

如果模型只知道 token 是什么，不知道 token 在第几个位置，就会丢掉顺序信息。

所以我们需要 position embedding。

它的作用是告诉模型：

```text
这个 token 在序列的第几个位置
```

---

## token embedding 是什么？

token embedding table 的形状是：

```text
[vocab_size, n_embd]
```

每个 token id 都会查出一个向量。

输入：

```text
idx: [batch, seq_len]
```

查表后：

```text
token_emb: [batch, seq_len, n_embd]
```

它回答的是：

```text
这个 token 是谁？
```

---

## position embedding 是什么？

position embedding table 的形状是：

```text
[block_size, n_embd]
```

每个位置都有一个可学习向量。

例如位置序列：

```text
[0, 1, 2, 3, ..., seq_len-1]
```

查表后：

```text
pos_emb: [seq_len, n_embd]
```

它回答的是：

```text
这个 token 在哪里？
```

---

## 为什么 token embedding 和 position embedding 可以相加？

token embedding 是：

```text
[batch, seq_len, n_embd]
```

position embedding 是：

```text
[seq_len, n_embd]
```

position embedding 会自动广播到 batch 维度。

相加后：

```text
x = token_emb + pos_emb
```

得到：

```text
[batch, seq_len, n_embd]
```

这个向量同时包含：

```text
token 是谁
token 在哪里
```

可以简单理解为：

```text
最终输入向量 = token 信息 + 位置信息
```

---

## logits 的 shape

经过语言模型头之后：

```text
x: [batch, seq_len, n_embd]
```

会变成：

```text
logits: [batch, seq_len, vocab_size]
```

含义是：

```text
每条序列
每个位置
都输出 vocab_size 个分数
```

这些分数用来预测下一个 token。

---

## 这节模型和 Bigram 有什么区别？

Bigram 模型直接是：

```text
token id -> vocab logits
```

这一课的模型是：

```text
token id
↓
token embedding
+
position embedding
↓
hidden vector
↓
vocab logits
```

所以它多了：

```text
位置感知能力
```

但它仍然没有 self-attention。

---

## 这节模型为什么还不是真正的 GPT？

因为每个位置虽然知道：

```text
自己是什么 token
自己在哪个位置
```

但它还不能真正读取前面其他 token 的信息。

真正的 GPT 需要 self-attention，让每个 token 可以看见它前面的上下文。

也就是说，本节模型还缺少：

```text
token 之间的信息交流
```

下一课 self-attention 就是为了解决这个问题。

---

## 这节课的核心结论

GPT 的输入不是只有 token embedding。

它通常是：

```text
token embedding + position embedding
```

token embedding 提供：

```text
这个 token 是谁
```

position embedding 提供：

```text
这个 token 在哪里
```

两者相加后，模型才知道序列中的内容和顺序。

但仅有位置还不够。

下一步需要：

```text
self-attention
```

让 token 之间真正互相读取信息。
