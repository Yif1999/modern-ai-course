# 08：CNN 收尾与 CIFAR-10 初体验

## 这一课解决什么问题？

前面我们用 MNIST 学习了 CNN。

MNIST 的特点是：

```text
28×28 灰度图
数字基本居中
背景干净
类别简单
```

所以即使 MLP 把图片直接 flatten 成 784 维，也能取得很不错的准确率。

这一课我们用 CIFAR-10 做对比，理解为什么更复杂的图片任务更能体现 CNN 的价值。

---

## MNIST 和 CIFAR-10 的区别

MNIST 图片形状是：

```text
[1, 28, 28]
```

含义是：

```text
1 个灰度通道
28 像素高
28 像素宽
```

CIFAR-10 图片形状是：

```text
[3, 32, 32]
```

含义是：

```text
3 个颜色通道：RGB
32 像素高
32 像素宽
```

所以 CIFAR-10 的输入更复杂。

---

## CIFAR-10 为什么更难？

CIFAR-10 的类别包括：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

它比 MNIST 更难，原因包括：

1. 图片是彩色的。
2. 背景更复杂。
3. 物体位置和姿态变化更大。
4. 类别之间差异不一定像数字那么明显。
5. 一些类别本身就比较相似，例如 cat 和 dog、automobile 和 truck。

所以一个很小的 CNN 只训练几轮，accuracy 通常不会像 MNIST 那样很快接近 0.98。

---

## 本节 CNN 结构

本节模型结构是：

```text
输入：[batch, 3, 32, 32]
↓
Conv2d: 3 → 16
↓
ReLU
↓
MaxPool: 32×32 → 16×16
↓
Conv2d: 16 → 32
↓
ReLU
↓
MaxPool: 16×16 → 8×8
↓
Flatten: 32×8×8 = 2048
↓
Linear: 2048 → 10
```

最后输出 10 个 logits，对应 CIFAR-10 的 10 个类别。

---

## 为什么第一层卷积是 3 → 16？

MNIST 是灰度图，所以第一层通常是：

```python
Conv2d(1, 8, ...)
```

CIFAR-10 是 RGB 彩色图，所以输入通道数是 3。

因此第一层写成：

```python
Conv2d(3, 16, ...)
```

意思是：

```text
从 3 个颜色通道中提取 16 张 feature maps
```

---

## conv1.weight 为什么是 [16, 3, 3, 3]？

Conv2d 的权重形状是：

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

所以：

```python
Conv2d(3, 16, kernel_size=3)
```

对应：

```text
[16, 3, 3, 3]
```

含义是：

```text
16：输出通道数
3：输入通道数，也就是 RGB
3：卷积核高度
3：卷积核宽度
```

---

## 这节课的核心结论

MNIST 上 MLP 和 CNN 都表现不错，不代表 MLP 和 CNN 一样适合图片。

更准确的理解是：

```text
MNIST 太简单，所以 MLP 也能学得很好。
```

CNN 的优势在于：

```text
保留空间结构
关注局部特征
权重共享
更适合彩色和复杂图像
参数利用效率更高
```

CIFAR-10 比 MNIST 更复杂，所以一个小 CNN 的准确率不会轻松达到 MNIST 那么高。

这说明：

```text
任务越复杂，模型结构和特征提取方式越重要。
```

---

## 下一阶段预告

CNN 图像基础到这里基本完成。

接下来可以切换到语言模型主线：

```text
token
vocab
embedding
sequence
bigram model
self-attention
tiny GPT
```

也就是从图像分类过渡到 Transformer / GPT 实践。
