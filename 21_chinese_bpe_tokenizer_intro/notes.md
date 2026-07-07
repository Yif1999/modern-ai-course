# 21：中文 BPE / Subword Tokenizer 入门

## 这一课解决什么问题？

上一课我们把中文文本处理成字符级 token ids。

字符级 tokenizer 很直观：

```text
人工智能
↓
人 / 工 / 智 / 能
```

但真实语言模型通常不会只用字符级 tokenizer。

本节引入 BPE tokenizer，观察它如何把常见字符片段合并成更长的 subword token。

---

## 字符级 tokenizer 是什么？

字符级 tokenizer 把每个字符都当成一个 token。

优点：

1. 实现简单。
2. 不需要训练 tokenizer。
3. 中文不会出现未知词问题。
4. encode / decode 很容易检查。

缺点：

1. 序列更长。
2. 常见词语会被拆散。
3. 同样一段文本需要更多上下文位置。
4. 模型要自己从字符组合里学习词语和短语。

---

## BPE tokenizer 是什么？

BPE 全称是 Byte Pair Encoding。

在 NLP 里，它通常表示一种 subword tokenizer。

它的核心思想是：

```text
先从小单位开始
不断合并最常见的相邻片段
最后形成一个固定大小的 vocab
```

对于中文，可以理解为：

```text
人 / 工 / 智 / 能
```

如果 `人工智能` 经常出现，BPE 可能逐步学到：

```text
人工
智能
人工智能
```

---

## BPE 为什么从字符开始合并高频片段？

因为字符集合比较小，可以覆盖几乎所有文本。

然后 BPE 根据训练语料里的频率，优先合并经常一起出现的片段。

这样可以在“覆盖能力”和“压缩效率”之间折中。

字符级 tokenizer 覆盖能力强，但序列长。

BPE tokenizer 仍然能覆盖文本，同时可以把常见词语压缩成更少 token。

---

## 中文 BPE 可能学到哪些片段？

如果训练语料里经常出现这些词：

```text
人工智能
语言模型
tokenizer
Tiny GPT
训练数据
```

BPE 可能会把它们学成一个或几个 token。

这不是我们手写规则，而是 tokenizer 从语料统计里学出来的。

---

## vocab_size 越大通常会怎样？

`vocab_size` 越大，tokenizer 可以保存的片段越多。

通常会带来：

1. 平均 token 数减少。
2. 常见词更容易被合并成一个 token。
3. embedding table 变大。
4. LM Head 输出维度变大。

所以 vocab_size 不是越大越好。

它会影响：

```text
序列长度
模型参数量
训练速度
生成质量
```

---

## BPE tokenizer 和模型训练的关系

模型不直接读取字符串。

模型读取的是：

```text
token ids
```

所以 tokenizer 会决定：

1. 输入序列有多长。
2. vocab_size 是多少。
3. embedding table 有多少行。
4. LM Head 输出多少个 logits。

例如：

```text
vocab_size = 512
```

那么模型最后一层每个位置要输出：

```text
512 个 token 分数
```

---

## special tokens 的作用

本节加入了：

```text
<pad>
<unk>
<bos>
<eos>
```

含义是：

`<pad>`：补齐不同长度序列。

`<unk>`：表示未知 token。

`<bos>`：begin of sequence，序列开始。

`<eos>`：end of sequence，序列结束。

这些 token 在真实训练、batch 对齐、生成控制里很常见。

---

## encode / decode 为什么必须可逆？

`encode` 是：

```text
文本 -> token ids
```

`decode` 是：

```text
token ids -> 文本
```

如果 decode 不能还原原文，说明 tokenizer 会丢信息。

对语言模型来说，这会直接影响训练和生成调试。

所以本节报告里会检查：

```text
decoded == original text
```

---

## 中文、英文、数字、标点混排为什么更复杂？

真实文本经常是混排的：

```text
中文、English、数字123和标点符号！
```

这里同时包含：

1. 中文汉字。
2. 英文字母。
3. 数字。
4. 空格。
5. 中文标点。
6. 英文标点。

tokenizer 必须能稳定处理这些符号，并且 encode / decode 不丢失。

这也是为什么真实大模型会使用成熟 tokenizer，而不是简单手写规则。

---

## 这节课的核心结论

字符级 tokenizer 更简单，但序列更长。

BPE tokenizer 更复杂，但能把高频片段合并成更少 token。

本节要重点观察：

```text
同一段文本的 token 数是否下降
BPE vocab 是否出现常见中文片段
不同 vocab_size 对 token 数的影响
encode / decode 是否可逆
```

下一课可以把 BPE 输出的 token ids 接入 Tiny GPT，训练中文 BPE 版本的小语言模型。
