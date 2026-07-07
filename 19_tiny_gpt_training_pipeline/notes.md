# 19-21 综合课程：Tiny GPT 训练强化与工程化

## 这一课解决什么问题？

前面的 Tiny GPT 课程主要是单个脚本实验：能训练、能生成、能保存 loss 曲线。

这一课把它整理成一个更像真实项目的结构：

```text
prepare_dataset.py
tokenizer.py
dataset.py
model.py
train.py
generate.py
config.py
utils.py
```

重点不是让模型结构突然变强，而是让实验更清晰、可保存、可复现、可扩展。

---

## 数据 pipeline

数据流程是：

```text
data/raw/tiny_text.txt
↓
字符级 tokenizer
↓
token ids
↓
train / val split
↓
data/processed/tiny_text_processed.npz
```

`prepare_dataset.py` 负责把原始文本处理成 token id，并保存：

```text
train_ids
val_ids
vocab.json
dataset_meta.json
```

`dataset.py` 负责从 token ids 中采样 batch。

每个 batch 的形状是：

```text
x: [batch_size, block_size]
y: [batch_size, block_size]
```

其中 `y` 是 `x` 向右移动一位后的 next-token target。

---

## 模型结构

Tiny GPT 的数据流是：

```text
token ids
↓
token embedding
↓
position embedding
↓
Transformer Block × N
↓
final LayerNorm
↓
LM Head
↓
logits
```

每个 Transformer Block 包含：

```text
LayerNorm
Multi-Head Causal Self-Attention
Residual Connection
LayerNorm
FeedForward MLP
Residual Connection
```

block 内部输入输出 shape 都保持：

```text
[batch, seq_len, n_embd]
```

这样多个 block 才能连续堆叠。

---

## logits 和 loss

模型最终输出：

```text
logits: [batch, seq_len, vocab_size]
```

意思是：每个 batch 里每条序列的每个位置，都要预测下一个 token。

训练时使用 cross entropy：

```text
cross_entropy(logits, targets)
```

它会让正确的下一个 token 得到更高分数。

---

## 训练循环

`train.py` 的核心流程是：

```text
get_batch
↓
forward 得到 logits
↓
计算 next-token loss
↓
nn.value_and_grad 求梯度
↓
optimizer.update 更新参数
↓
定期评估 train / val loss
↓
保存 checkpoint、loss 曲线、生成样本
```

这就是一个最小但完整的语言模型训练工程。

---

## checkpoint 的作用

checkpoint 保存的是训练过程中的模型权重，以及尽量保存 optimizer 状态。

它的作用是：

1. 训练中断后可以继续训练。
2. 可以保存最好的一版模型。
3. 可以单独运行 `generate.py` 做推理。
4. 后续可以比较不同超参数实验。

本项目会维护：

```text
outputs/checkpoints/latest.json
outputs/checkpoints/best.json
```

分别指向最新 checkpoint 和最佳 validation loss checkpoint。

---

## loss 曲线怎么看？

`train loss` 表示模型在训练集上的拟合情况。

`val loss` 表示模型在没直接训练过的数据上的表现。

常见现象：

```text
train loss 下降，val loss 也下降：模型确实在学习
train loss 很低，val loss 明显变高：可能过拟合
两者都不下降：模型、数据或学习率可能有问题
```

---

## 生成样本的作用

loss 是数字指标，但语言模型最终要看生成文本。

所以训练过程中定期保存样本：

```text
outputs/samples/
```

这样可以观察：

1. 模型是否学到训练文本的风格。
2. 是否开始重复。
3. 是否只是机械记忆。
4. loss 下降和文本质量是否同步改善。

---

## 超参数实验

本项目把超参数集中放在 `config.py`：

```text
block_size
batch_size
n_embd
num_heads
num_layers
learning_rate
max_iters
```

以后做实验时，可以一次只改一个参数，然后比较：

```text
loss 曲线
best val loss
生成样本
训练速度
```

这比在一个大脚本里到处改数字更可靠。

---

## 这节课的核心结论

这一课不是新的模型结构课，而是训练工程课。

核心变化是：

```text
单脚本 demo
↓
可复用 Tiny GPT 训练项目
```

现在项目已经具备：

1. 数据准备。
2. tokenizer。
3. batch 采样。
4. Tiny GPT 模型。
5. 训练入口。
6. 生成入口。
7. checkpoint。
8. loss 曲线。
9. 训练日志。
10. 定期样本。

后面如果要进入更真实的数据集、更复杂 tokenizer、更大模型或更正式的训练流程，就可以在这个项目骨架上扩展。
