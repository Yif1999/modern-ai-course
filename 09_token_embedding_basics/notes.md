# 09：Token / Vocab / Embedding 基础

## 这一课解决什么问题？

前面我们学习图像模型时，输入是像素。

例如 MNIST 输入是：

```text
[batch, 1, 28, 28]
```

CNN 会把像素变成 feature maps，再输出分类 logits。

进入语言模型后，输入不再是像素，而是文本。

但是模型不能直接处理字符串，所以我们要先把文本变成数字。

语言模型的基础流程是：

```text
文本
↓
token
↓
token id
↓
embedding 向量
↓
logits
↓
预测下一个 token
```

## 什么是 token？

token 是文本被切分后的基本单位。

本节为了简单，使用字符级 tokenization。

例如：

```text
hello
```

会被拆成：

```text
h, e, l, l, o
```

每个字符就是一个 token。

真实大语言模型通常不是按字符切，而是使用更复杂的 tokenizer，例如 BPE、SentencePiece 等。

但核心思想一样：

```text
文本要先被切成 token，然后每个 token 映射成一个整数 id。
```

## 什么是 vocab？

vocab 是词表，也就是所有 token 的集合。

例如文本中出现过这些字符：

```python
['\n', ' ', 'a', 'b', 'e', 'g', 'h', 'i', 'l', 'm', 'o', 'p', 't', 'x', 'y']
```

每个 token 都会分配一个编号。

例如：

```text
'h' → 6
'e' → 4
'l' → 8
'o' → 10
```

这个映射叫：

```text
stoi: string to index
```

反过来：

```text
6 → 'h'
4 → 'e'
8 → 'l'
10 → 'o'
```

这个映射叫：

```text
itos: index to string
```

## encode 和 decode

encode 的作用是：

```text
文本 → token ids
```

例如：

```text
"hello" → [6, 4, 8, 8, 10]
```

decode 的作用是：

```text
token ids → 文本
```

例如：

```text
[6, 4, 8, 8, 10] → "hello"
```

这两个函数是语言模型输入输出的基础。

## 什么是 next-token prediction？

语言模型最核心的训练目标是：

```text
根据前面的 token，预测下一个 token。
```

例如原始文本是：

```text
hello
```

输入序列可以是：

```text
h e l l
```

目标序列就是：

```text
e l l o
```

也就是说：

```text
看到 h，预测 e
看到 h e，预测 l
看到 h e l，预测 l
看到 h e l l，预测 o
```

所以训练数据里：

```text
x_batch 是当前 token 序列
y_batch 是向右移动一位的下一个 token 序列
```

## 输入 shape：[batch, seq_len]

语言模型输入通常是：

```text
[batch, seq_len]
```

含义是：

```text
batch：一次训练多少条序列
seq_len：每条序列有多少个 token
```

例如：

```text
[4, 8]
```

表示：

```text
一次输入 4 条序列，每条序列长度是 8 个 token。
```

每个位置存的是一个整数 token id。

## 什么是 embedding？

token id 本身只是一个整数。

例如：

```text
'h' → 6
```

但是模型不能把数字 6 当成真正的语义向量。

所以我们需要 embedding table。

embedding table 可以理解为一张查表矩阵：

```text
[vocab_size, embed_dim]
```

例如：

```text
[15, 4]
```

表示：

```text
词表里有 15 个 token
每个 token 对应一个 4 维向量
```

当输入是：

```text
[batch, seq_len]
```

查 embedding 后会变成：

```text
[batch, seq_len, embed_dim]
```

这一步相当于把离散的 token id 变成连续向量。

## 什么是 logits？

语言模型最终要预测下一个 token。

如果 vocab_size 是 15，那么每个位置都要输出 15 个分数：

```text
每个分数对应一个 token 成为下一个 token 的可能性。
```

所以 logits 的 shape 是：

```text
[batch, seq_len, vocab_size]
```

例如：

```text
[4, 8, 15]
```

表示：

```text
4 条序列
每条序列 8 个位置
每个位置输出 15 个 token 分数
```

## Cross Entropy 在预测什么？

这里的 cross entropy 不是预测图片类别，而是在预测下一个 token。

输入：

```text
logits: [batch, seq_len, vocab_size]
```

目标：

```text
y_batch: [batch, seq_len]
```

每个位置的目标都是正确的下一个 token id。

所以训练目标是：

```text
让模型在每个位置给正确的下一个 token 更高分数。
```

这和图片分类本质上是一样的：

```text
图片分类：从 10 个类别里选正确类别
语言模型：从 vocab_size 个 token 里选正确的下一个 token
```

## PyTorch 和 MLX 的共同点

PyTorch 中：

```python
embedding = torch.nn.Embedding(vocab_size, embed_dim)
token_embeddings = embedding(x_batch)
```

MLX 中我们手写：

```python
embedding_table = mx.random.normal((vocab_size, embed_dim))
token_embeddings = embedding_table[x_batch]
```

本质都一样：

```text
用 token id 去 embedding table 里查对应向量。
```

## 这节课的核心结论

语言模型不是直接吃字符串。

它吃的是：

```text
整数 token id
```

然后通过 embedding table 变成：

```text
连续向量
```

再输出：

```text
每个位置对下一个 token 的 logits
```

所以从这一课开始，我们正式进入 GPT 的底层结构：

```text
token ids
↓
embedding
↓
logits
↓
next-token prediction
```

后面的 bigram model、self-attention、Tiny GPT，都会建立在这套输入输出格式上。
