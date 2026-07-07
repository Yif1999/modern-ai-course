# 第 9 课：CNN Filters / Feature Maps 可视化

## 这一课解决什么问题？

上一课我们理解了 CNN 的基本结构：

```text
Conv2d -> ReLU -> Pool -> Conv2d -> ReLU -> Pool -> Flatten -> Linear
```

这一课进一步观察 CNN 中间到底在做什么。

核心概念有两个：

```text
filter / kernel：卷积核，是模型学出来的参数
feature map：特征图，是卷积核扫过输入后得到的输出
```

## 什么是卷积核？

卷积核是卷积层里的可训练参数。

例如：

```python
Conv2d(1, 8, kernel_size=3, padding=1)
```

表示创建 8 个 `3 x 3` 的卷积核。

因为输入通道数是 1，所以权重形状是：

```text
[8, 1, 3, 3]
```

含义是：

```text
8：输出通道数，也就是 8 个卷积核
1：输入通道数
3：卷积核高度
3：卷积核宽度
```

这些卷积核的具体数值不是我们手写的，而是训练过程中通过反向传播学出来的。

## 什么是 feature map？

feature map 是卷积核扫过输入后得到的响应图。

一张输入图片经过 8 个卷积核，就会得到 8 张 feature maps。

可以理解为：

```text
第 0 个卷积核看到的图案响应
第 1 个卷积核看到的图案响应
第 2 个卷积核看到的图案响应
...
```

不同 feature map 可能对不同局部模式更敏感，例如边缘、笔画、拐角、局部纹理等。

## filter 和 feature map 的区别

filter / kernel 是：

```text
模型参数
```

feature map 是：

```text
模型中间输出
```

更直白地说：

```text
filter 是“检测器”
feature map 是“检测结果”
```

卷积核负责检测某种局部模式，feature map 显示这种模式在图片哪些位置响应更强。

## 为什么一张图会产生多张 feature maps？

因为一个卷积层通常不只有一个卷积核。

例如：

```python
Conv2d(1, 8, kernel_size=3)
```

有 8 个输出通道，也就是 8 个卷积核。

同一张图片被 8 个不同卷积核分别扫描，所以产生 8 张 feature maps。

## 第二层卷积为什么是 [16, 8, 3, 3]？

第一层卷积输出 8 个通道。

所以第二层卷积的输入不再是一张灰度图，而是 8 张 feature maps。

```python
Conv2d(8, 16, kernel_size=3)
```

权重形状是：

```text
[16, 8, 3, 3]
```

含义是：

```text
16：输出通道数
8：输入通道数
3 x 3：每个输入通道上的局部窗口
```

第二层会把 8 个输入通道的信息组合起来，生成 16 个更高级的 feature maps。

## 这节课的核心结论

CNN 的卷积层不是手写规则。

我们只指定：

```text
卷积核大小
输入通道数
输出通道数
padding
```

卷积核里的具体数字由训练学出来。

训练后：

```text
filter / kernel：学到的局部模式检测器
feature map：某个检测器对输入图片的响应结果
```

所以 CNN 可以逐层从简单局部特征，组合出更复杂的图像表示。
