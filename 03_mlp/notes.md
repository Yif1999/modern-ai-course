# 第 5 课：MLP，多层感知机

## 这一课解决什么问题？

前面我们训练的是线性回归模型：

```text
y_pred = w * x + b
```

它只能表达一条直线。

但很多真实问题不是线性的。例如本节课里的目标函数：

```text
y = x²
```

这是一条曲线，单纯的线性模型很难拟合好。

所以我们需要 MLP，也就是多层感知机。

---

## MLP 的基本结构

一个最简单的 MLP 可以写成：

```text
输入
↓
线性层
↓
非线性激活函数
↓
线性层
↓
输出
```

本节课的结构是：

```text
1 → 16 → 1
```

意思是：

- 输入维度是 1。
- 隐藏层维度是 16。
- 输出维度是 1。

---

## 线性层是什么？

线性层本质上还是：

```text
y = xW + b
```

只不过参数不再是一个标量，而是一组矩阵和向量。

例如：

```text
Linear(1, 16)
```

表示：把 1 维输入映射成 16 维隐藏表示。

然后：

```text
Linear(16, 1)
```

表示：把 16 维隐藏表示映射回 1 维输出。

---

## ReLU 为什么重要？

ReLU 是一种非线性激活函数：

```text
ReLU(x) = max(0, x)
```

它的作用是给模型加入非线性能力。

如果没有 ReLU，那么：

```text
Linear
↓
Linear
↓
Linear
```

无论叠多少层，本质上仍然等价于一个大的线性变换。

也就是说，没有非线性激活函数，多层网络并不会真正变强。

加入 ReLU 后，模型就可以拟合更复杂的函数，例如：

```text
y = x²
```

---

## 为什么 MLP 可以拟合曲线？

可以把隐藏层里的 16 个神经元理解成 16 个“可学习的小特征”。

每个神经元都先做一次线性变换，再经过 ReLU。

这些经过 ReLU 的特征组合起来，就可以拼出更复杂的形状。

所以 MLP 不只是学一条直线，而是可以用很多段非线性片段去逼近曲线。

---

## PyTorch 写法

PyTorch 中可以用：

```python
torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
)
```

这表示按顺序执行：

```text
Linear → ReLU → Linear
```

训练循环仍然是：

```python
y_pred = model(x)
loss = ...
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## MLX 写法

MLX 中可以用：

```python
self.linear1 = nn.Linear(1, 16)
self.linear2 = nn.Linear(16, 1)
```

然后在 `__call__` 中手动写：

```python
h = self.linear1(x)
h = nn.relu(h)
y = self.linear2(h)
```

训练循环仍然是：

```python
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y_true)
optimizer.update(model, grads)
mx.eval(loss, model.parameters(), optimizer.state)
```

---

## 这节课的核心结论

MLP 的强大不只是因为“层数变多”。

真正关键的是：

```text
线性变换 + 非线性激活函数
```

线性层负责学习特征组合，非线性激活函数让模型能够表达曲线和复杂模式。

从这一课开始，我们已经进入真正的神经网络世界。

后面的 CNN、Transformer、GPT、Diffusion，本质上都是在这个基础上继续堆更复杂的模块。
