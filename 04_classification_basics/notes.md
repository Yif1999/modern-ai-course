# 第 6 课：分类任务、Logits、Softmax、Cross Entropy

## 这一课解决什么问题？

前几课我们做的是回归任务。

回归任务预测的是连续数值，例如：

```text
y = x²
```

输入一个 `x`，模型输出一个具体数值。

但很多 AI 任务是分类任务，例如：

- 图片是猫还是狗
- 手写数字是 0 到 9 中的哪一个
- 一句话表达的是正面还是负面情绪

分类任务预测的不是连续数值，而是“类别”。

---

## 本节任务

我们构造了一个二维分类任务：

```text
如果 x1 + x2 > 0，则类别为 1
否则类别为 0
```

模型输入是二维向量：

```text
[x1, x2]
```

输出是两个类别的分数：

```text
[class_0_score, class_1_score]
```

---

## 什么是 logits？

logits 是模型最后一层直接输出的原始分数。

例如一个二分类模型可能输出：

```text
[1.2, 3.5]
```

这不是概率，因为它们：

1. 不一定在 0 到 1 之间。
2. 加起来不一定等于 1。

但它们表示模型对每个类别的“偏好程度”。

数值越大，模型越倾向于选择那个类别。

---

## 什么是 softmax？

softmax 可以把 logits 转成概率分布。

例如：

```text
logits = [1.2, 3.5]
```

softmax 后可能变成：

```text
probs = [0.09, 0.91]
```

这表示模型认为：

- 类别 0 的概率约为 9%
- 类别 1 的概率约为 91%

分类预测通常取概率最大的类别。

---

## 什么是 Cross Entropy？

Cross Entropy 是分类任务常用的 loss。

它会惩罚错误分类，尤其会强烈惩罚“自信但错误”的预测。

例如真实类别是 1：

- 正确但不自信：类别 1 概率 0.6，loss 较低
- 正确且自信：类别 1 概率 0.99，loss 更低
- 错误且自信：类别 1 概率 0.01，loss 很高

所以 Cross Entropy 会推动模型：

- 给正确类别更高分数
- 给错误类别更低分数

---

## 为什么训练时直接传 logits？

在 PyTorch 和 MLX 里，cross entropy 通常直接接收 logits。

原因是：

```text
cross entropy 内部会处理 softmax / log-softmax
```

这样做数值上更稳定，也更方便。

所以训练时通常写：

```python
loss = cross_entropy(logits, labels)
```

而不是：

```python
probs = softmax(logits)
loss = cross_entropy(probs, labels)
```

---

## 什么是 accuracy？

accuracy 是准确率。

计算方式是：

```text
预测正确的样本数 / 总样本数
```

例如 100 个样本里预测对了 95 个：

```text
accuracy = 0.95
```

loss 用于训练，accuracy 用于观察模型表现。

---

## PyTorch 写法

PyTorch 分类任务常见流程：

```python
logits = model(x)
loss = torch.nn.functional.cross_entropy(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
predictions = torch.argmax(logits, dim=1)
accuracy = (predictions == y).float().mean()
```

其中：

```text
dim=1
```

表示在类别维度上取最大值。

---

## MLX 写法

MLX 分类任务常见流程：

```python
logits = model(x)
loss = nn.losses.cross_entropy(logits, y, reduction="mean")
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
optimizer.update(model, grads)
predictions = mx.argmax(logits, axis=1)
accuracy = mx.mean((predictions == y).astype(mx.float32))
```

其中：

```text
axis=1
```

表示在类别维度上取最大值。

---

## 这节课的核心结论

分类模型最后输出的不是一个数值，而是一组类别分数。

这组分数叫：

```text
logits
```

训练时使用：

```text
cross entropy
```

观察预测结果时，可以用：

```text
softmax
```

把 logits 转成概率。

最终预测类别时，通常使用：

```text
argmax
```

选择分数最高的类别。

这套流程会在 MNIST、图像分类、文本分类、LLM 的 token 预测中反复出现。
