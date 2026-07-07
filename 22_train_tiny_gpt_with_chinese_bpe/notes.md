# 22：用中文 BPE Token 训练 Tiny GPT

## 这一课解决什么问题？

前面已经完成两件事：

1. 第 20 课：中文 raw text 到字符级数据管线。
2. 第 21 课：训练中文 BPE tokenizer，并观察 BPE 和字符级 tokenizer 的区别。

本节把 BPE tokenizer 接入 Tiny GPT 训练流程。

目标是打通：

```text
中文文本
↓
BPE tokenizer
↓
BPE token ids
↓
Tiny GPT
↓
next-token loss
↓
中文生成
```

---

## 为什么从字符级 tokenizer 切换到 BPE tokenizer？

字符级 tokenizer 把每个字符当成一个 token。

例如：

```text
人工智能
↓
人 / 工 / 智 / 能
```

BPE tokenizer 会尝试把高频片段合并起来。

例如：

```text
人工智能
↓
人工智能
```

或者：

```text
语言模型
↓
语言模型
```

这样同样的文本通常会变成更短的 token 序列。

---

## 中文 BPE tokenizer 的基本流程

本节优先复用第 21 课训练好的：

```text
chinese_bpe_vocab512.json
```

如果找不到旧 tokenizer，就在本课用中文语料重新训练一个小型 BPE tokenizer。

流程是：

```text
读取中文 raw text
↓
加载或训练 BPE tokenizer
↓
encode 成 BPE token ids
↓
加入 <bos> / <eos>
↓
划分 train / val
↓
保存 .npy token 数据
```

---

## BPE token ids 如何进入 Tiny GPT？

模型输入仍然是：

```text
x: [batch, seq_len]
```

只是这里的整数不再表示“字符 id”，而是表示“BPE token id”。

Tiny GPT 做的事情不变：

```text
token ids
↓
token embedding
↓
position embedding
↓
Transformer Blocks
↓
LM Head
↓
logits
```

---

## 数据 shape 上的相同点

无论字符级还是 BPE，训练 batch 的形状都是：

```text
x: [batch_size, block_size]
y: [batch_size, block_size]
```

其中：

```text
y = x 向右移动一位
```

cross entropy 预测的都是：

```text
下一个 token
```

---

## vocab_size / seq_len 上的差异

字符级 tokenizer 的特点：

```text
vocab_size 较小
seq_len 较长
```

BPE tokenizer 的特点：

```text
vocab_size 较大
seq_len 较短
```

这会影响模型：

1. `vocab_size` 变大，embedding table 变大。
2. `vocab_size` 变大，LM Head 输出维度变大。
3. `seq_len` 变短，同样的 `block_size` 可以覆盖更多原始文字。

---

## logits shape 为什么是 [batch, seq_len, vocab_size]？

语言模型在每个位置都要预测下一个 token。

如果：

```text
batch = 32
seq_len = 64
vocab_size = 512
```

那么 logits 是：

```text
[32, 64, 512]
```

含义是：

```text
32 条序列
每条序列 64 个位置
每个位置输出 512 个 token 分数
```

---

## loss 曲线观察

训练时需要同时看：

```text
train loss
val loss
```

如果二者都下降，说明模型确实在学习数据规律。

如果 train loss 很低，而 val loss 明显高，通常说明小模型在小语料上开始记忆训练文本。

---

## 中文生成样本观察

生成时模型输出的是 token id。

这些 id 本身不可读，所以必须经过：

```text
tokenizer.decode(ids)
```

还原成中文文本。

当前实验语料很小，模型也很小，所以生成文本可能会重复、不自然，甚至混杂训练语料片段。

这不是 BPE 的问题，而是数据规模和模型规模都很小。

---

## 当前实验的局限

1. 中文语料很小。
2. tokenizer 也是在小语料上训练的。
3. Tiny GPT 参数量很小。
4. 训练主要是教学实验，不追求真实生成质量。
5. 小语料很容易过拟合。

---

## 下一步

下一步可以进入更真实的中文短文本预训练实验：

```text
更大 raw text
更稳定的数据清洗
更合理的 train / val split
BPE token 数据缓存
Tiny GPT 中文训练对比实验
```

也就是从“管线打通”走向“更像真实预训练实验”。
