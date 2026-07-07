# 第 7 课：MNIST 手写数字分类

## 这一课解决什么问题？

前一课我们做了一个二维点分类任务：

```text
[x1, x2] -> 0 或 1
```

这一课我们进入图片分类：

```text
手写数字图片 -> 0 到 9 中的一个类别
```

MNIST 是深度学习里非常经典的入门数据集。

每张图片是：

```text
28 x 28
```

的灰度图，所以一共有：

```text
28 x 28 = 784
```

个像素值。

## 图片如何进入神经网络？

MLP 接收的是一维向量，不直接接收二维图片。

所以我们要把图片：

```text
[1, 28, 28]
```

展平成：

```text
[784]
```

如果有一个 batch，例如 128 张图片，那么输入形状是：

```text
[128, 784]
```

其中：

- 128 是 batch size。
- 784 是每张图片的像素特征数量。

## MNIST 是 10 类分类

MNIST 的标签是：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

所以模型最后不能只输出一个数，而要输出 10 个分数：

```text
[class_0_score, class_1_score, ..., class_9_score]
```

这 10 个原始分数叫：

```text
logits
```

模型预测时，选择 logits 最大的那个类别：

```text
argmax(logits)
```

## 本节 MLP 结构

本节使用的模型结构是：

```text
784 -> 128 -> 10
```

含义是：

1. 输入层：784 个像素值。
2. 隐藏层：128 个神经元。
3. 输出层：10 个类别分数。

中间使用 ReLU 激活函数：

```text
Linear -> ReLU -> Linear
```

## Loss：Cross Entropy

因为 MNIST 是分类任务，所以使用 cross entropy loss。

训练时直接传入 logits 和正确标签：

```python
loss = cross_entropy(logits, labels)
```

不需要手动 softmax，因为框架内部会用更稳定的方式处理。

## Accuracy：准确率

accuracy 的计算方式是：

```text
预测正确的样本数 / 总样本数
```

例如：

```text
10000 张测试图片，预测对 9700 张
```

则：

```text
accuracy = 0.97
```

loss 用来优化模型，accuracy 用来观察模型实际分类表现。

## Mini-batch 训练

MNIST 训练集有 60000 张图片。

如果一次把所有图片都拿来训练，会比较重，也不符合常见训练方式。

所以我们使用 mini-batch：

```text
每次拿 128 张图片训练一步
```

这样每一步的输入形状是：

```text
[128, 784]
```

每一步的标签形状是：

```text
[128]
```

每个 epoch 会遍历完整训练集一次。

## PyTorch 版本

PyTorch 版本使用：

- `torchvision.datasets.MNIST`
- `DataLoader`
- `torch.nn.Module`
- `torch.optim.Adam`
- `torch.nn.functional.cross_entropy`

本课的数据保存在课程目录内：

```text
05_mnist/data
```

这样不会污染整个 AI Lab 项目根目录。

训练流程是：

```text
取一个 batch
图片移动到 MPS
模型输出 logits
计算 cross entropy loss
loss.backward()
optimizer.step()
统计 accuracy
```

## MLX 版本

MLX 版本使用：

- `torchvision` 下载数据
- `numpy` 整理数据
- `mx.array` 转成 MLX 数组
- `mlx.nn.Module`
- `mlx.optimizers.Adam`
- `nn.losses.cross_entropy`

训练流程是：

```text
取一个 batch
模型输出 logits
计算 cross entropy loss
nn.value_and_grad 求梯度
optimizer.update 更新模型
mx.eval 触发计算
统计 accuracy
```

## PyTorch 和 MLX 的共同点

两者本质都在做同一件事：

```text
图片像素 -> MLP -> logits -> cross entropy -> 梯度 -> 更新参数
```

共同核心是：

- forward
- loss
- gradient
- optimizer
- accuracy

## PyTorch 和 MLX 的区别

PyTorch 更像是：

```text
loss.backward()
梯度存到参数里
optimizer.step()
```

MLX 更像是：

```text
value_and_grad(loss_fn)
直接返回梯度
optimizer.update(model, grads)
```

PyTorch 需要显式把模型和数据放到：

```text
mps
```

MLX 依赖 Apple Silicon 的统一内存和惰性计算，通常通过：

```text
mx.eval(...)
```

触发执行。

## 这节课的核心结论

MNIST 并没有引入新的训练本质。

它只是把上一节的分类任务从：

```text
二维点分类
```

升级成：

```text
图片分类
```

输入从：

```text
[x1, x2]
```

变成：

```text
784 个像素值
```

输出从：

```text
2 个类别 logits
```

变成：

```text
10 个类别 logits
```

训练逻辑仍然是：

```text
logits -> cross entropy -> gradient -> optimizer
```

这就是图像分类的最小完整流程。
