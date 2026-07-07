# 10：Bigram 语言模型

## 这一课解决什么问题？

上一课我们学习了：

```text
token
vocab
encode / decode
embedding
logits
next-token prediction
```

这一课开始真正训练一个语言模型。

不过它还不是 GPT，也没有 self-attention。

它是最简单的 Bigram 语言模型。

---

## 什么是 Bigram？

Bigram 的意思是：

```text
两个连续 token 之间的关系
```

在本节课里，模型只根据当前 token 预测下一个 token。

例如：

```text
h -> e
e -> l
l -> l
l -> o
```

它不会真正理解更长的上下文。

例如看到：

```text
hello ai
```

它不会综合整个句子，只会看最后一个 token 去预测下一个 token。

---

## Bigram 模型的输入输出

输入是 token ids：

```text
idx: [batch, seq_len]
```

例如：

```text
[16, 8]
```

表示一次输入 16 条序列，每条序列有 8 个 token。

模型输出：

```text
logits: [batch, seq_len, vocab_size]
```

含义是：

```text
每条序列
每个位置
都输出一组 vocab_size 维的分数
```

这些分数表示：

```text
在当前位置，下一个 token 可能是谁
```

---

## 为什么 Embedding 是 [vocab_size, vocab_size]？

上一课 embedding table 是：

```text
[vocab_size, embed_dim]
```

表示把 token id 转成 embed_dim 维向量。

但 Bigram 模型更简单。

它直接让每个 token id 查出一组预测下一个 token 的 logits。

所以这里的表是：

```text
[vocab_size, vocab_size]
```

含义是：

```text
每个当前 token
都有一行 vocab_size 维分数
用来预测下一个 token
```

所以 Bigram 模型本质上可以理解成一张表：

```text
当前 token -> 下一个 token 的分数分布
```

---

## 为什么要 reshape logits 和 targets？

模型输出是：

```text
logits: [batch, seq_len, vocab_size]
```

目标是：

```text
targets: [batch, seq_len]
```

cross entropy 需要的常见形式是：

```text
logits: [N, vocab_size]
targets: [N]
```

所以要把 batch 和 seq_len 合并：

```text
[batch, seq_len, vocab_size]
->
[batch * seq_len, vocab_size]
```

targets 同理：

```text
[batch, seq_len]
->
[batch * seq_len]
```

这样每个位置都变成一个独立的分类任务：

```text
根据当前 token，预测正确的下一个 token
```

---

## 训练目标

训练时使用 cross entropy。

目标是：

```text
让模型在每个位置给正确的下一个 token 更高分数
```

这和图片分类非常像：

```text
图片分类：从 10 个类别里选正确类别
语言模型：从 vocab_size 个 token 里选正确的下一个 token
```

区别只是类别从图片类别变成了 token 类别。

---

## generate 是怎么生成文本的？

生成时从一个起始 token 开始。

例如：

```text
h
```

模型预测下一个 token。

得到下一个 token 后，把它接到序列后面：

```text
h e
```

再继续预测：

```text
h e l
```

不断重复，就可以生成一段文本。

但 Bigram 模型只看当前最后一个 token，所以生成效果通常比较乱。

---

## Bigram 模型缺什么？

Bigram 模型最大的问题是：

```text
没有上下文理解能力
```

它只知道：

```text
当前 token 后面经常跟什么
```

但不知道更长文本里发生了什么。

例如它不能真正理解：

```text
hello tiny gpt
```

里面前后 token 的长期关系。

真正的 GPT 会加入：

```text
token embedding
position embedding
self-attention
MLP
Transformer block
```

其中 self-attention 会让每个 token 能看见前面的上下文。

---

## PyTorch 和 MLX 的共同点

PyTorch 写法：

```python
logits = self.token_embedding_table(idx)
loss = cross_entropy(logits_flat, targets_flat)
loss.backward()
optimizer.step()
```

MLX 写法：

```python
logits = self.token_embedding_table[idx]
loss, grads = nn.value_and_grad(model, loss_fn)(model, idx, targets)
optimizer.update(model, grads)
```

本质都一样：

```text
token ids
↓
查表得到 logits
↓
cross entropy
↓
梯度更新表里的参数
```

---

## 这节课的核心结论

Bigram 是最小语言模型。

它已经具备语言模型训练的核心形式：

```text
输入 token ids
↓
输出每个位置的 next-token logits
↓
用 cross entropy 训练
↓
通过采样生成文本
```

但它还没有真正的上下文能力。

下一步我们会开始加入更强的结构：

```text
上下文
位置
self-attention
Transformer
Tiny GPT
```
