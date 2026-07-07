# 20：中文开源文本预训练数据管线

## 这一课解决什么问题？

前面的 Tiny GPT 主要使用英文 toy text。

如果后面要训练中文 Tiny GPT，第一步不是改模型，而是先把中文文本整理成稳定的数据格式。

本节只做数据管线，不训练模型。

目标是把：

```text
raw 中文文本
```

处理成：

```text
train_tokens.npy
val_tokens.npy
vocab.json
metadata.json
```

这些文件后续可以直接被 Tiny GPT 或 BPE tokenizer 实验复用。

---

## 数据目录规范

本课所有数据都放在：

```text
20_chinese_open_text_pretraining_dataset/data
```

其中：

```text
data/raw/input_zh.txt
```

保存原始中文文本。

```text
data/processed/
```

保存清洗后的语料、token ids、词表和元数据。

不要把数据放到项目根目录的 `./data`，否则不同课程会互相污染。

---

## 数据清洗流程

脚本 `prepare_chinese_dataset.py` 做了这些事情：

1. 如果 `data/raw/input_zh.txt` 不存在，自动创建一个小型中文示例语料。
2. 读取 raw text。
3. 统一换行符。
4. 做 Unicode 规范化。
5. 去掉空行。
6. 过滤过短文本。
7. 过滤过长文本。
8. 过滤中文比例太低的文本。
9. 删除重复段落。
10. 输出清洗后的 `clean_corpus_zh.txt`。

清洗不是为了让文本变得“完美”，而是让数据更稳定、更可检查。

---

## train / val split 的意义

训练集 `train_tokens.npy` 用来更新模型参数。

验证集 `val_tokens.npy` 不参与参数更新，只用来观察模型是否真正学到了规律。

如果 train loss 降低，而 val loss 明显更高，通常说明模型可能在记忆训练文本，也就是过拟合。

本节只是小规模示例，所以 val 只能作为流程检查。真实实验应该使用更大、更独立的验证文本。

---

## vocab / stoi / itos

本节使用字符级 tokenizer。

`vocab` 是清洗后文本中所有出现过的字符集合。

例如中文字符、标点、换行、空格都可能是 token。

`stoi` 表示：

```text
string to index
字符 -> 整数 id
```

`itos` 表示：

```text
index to string
整数 id -> 字符
```

它们保存在：

```text
data/processed/vocab.json
```

---

## encode / decode

`encode` 的作用是：

```text
中文文本 -> token ids
```

例如：

```text
人工智能
```

会变成类似：

```text
[12, 45, 88, 31]
```

`decode` 的作用是反过来：

```text
token ids -> 中文文本
```

这两个过程必须能互相还原，否则后续训练和生成都会难以调试。

---

## block_size / batch_size

`block_size` 是每条训练序列的 token 长度。

例如：

```text
block_size = 32
```

表示模型每次最多看 32 个 token 的上下文。

`batch_size` 是一次取多少条训练序列。

例如：

```text
batch_size = 32
```

表示一次训练会并行处理 32 条短序列。

数据加载模块输出：

```text
x: [batch_size, block_size]
y: [batch_size, block_size]
```

其中 `y` 是 `x` 向右移动一位，用于 next-token prediction。

---

## 中文字符级 tokenizer 的优点

字符级 tokenizer 的优点是：

1. 实现简单。
2. 不需要训练分词器。
3. 不会遇到未知中文词。
4. 很适合教学和调试。

---

## 中文字符级 tokenizer 的缺点

缺点也很明显：

1. 序列更长。
2. 一个词会被拆成多个字。
3. 模型需要更多位置才能理解长词和短语。
4. 训练效率不如成熟的 BPE / SentencePiece tokenizer。

所以字符级 tokenizer 适合入门，但后续会继续进入 BPE tokenizer。

---

## 这节课的核心结论

语言模型训练前，必须先把文本变成稳定的 token ids。

本节得到的核心产物是：

```text
raw text
cleaned corpus
vocab
train token ids
val token ids
metadata
```

后续课程可以直接基于这些文件继续做：

```text
BPE tokenizer
Tiny GPT 中文训练
更大规模预训练数据管线
```
