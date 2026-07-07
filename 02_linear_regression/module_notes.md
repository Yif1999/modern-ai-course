# 第 4 课：模型模块封装参数

## 这一课解决什么问题？

前几节课里，我们直接创建参数：

```python
w = ...
b = ...
```

这在只有两个参数时可以接受。

但真实神经网络会有很多层、很多参数，例如：

- Embedding
- Attention
- MLP
- LayerNorm
- 输出层

如果所有参数都散落在外面，就会很难管理。

所以框架提供了“模型模块”：

- PyTorch: `torch.nn.Module`
- MLX: `mlx.nn.Module`

模型模块的作用是：

1. 把参数组织在一个对象里。
2. 把前向计算写成一个统一接口。
3. 让 optimizer 可以一次性拿到所有可训练参数。
4. 方便后续嵌套更复杂的层。

---

## PyTorch 的写法

PyTorch 中通常这样定义模型：

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b
```

关键点：

- `torch.nn.Module` 是所有模型的基类。
- `super().__init__()` 用来初始化模块内部的参数管理系统。
- `torch.nn.Parameter` 表示这个张量是需要训练的参数。
- `forward()` 定义模型如何从输入得到输出。
- 调用 `model(x)` 时，PyTorch 会自动调用 `model.forward(x)`。
- `model.parameters()` 会返回模型中所有需要训练的参数。
- `model.named_parameters()` 可以看到参数名字和参数值。

标准训练循环：

```python
y_pred = model(x)
loss = ...
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## MLX 的写法

MLX 中可以这样定义模型：

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.array(0.0)
        self.b = mx.array(0.0)

    def __call__(self, x):
        return self.w * x + self.b
```

关键点：

- `mlx.nn.Module` 是 MLX 中组织模型参数的基类。
- 在模块中保存的 `mx.array` 可以被模块参数系统管理。
- MLX 通常通过 `__call__` 定义模型调用方式。
- `model.parameters()` 可以返回模型参数结构。
- `nn.value_and_grad(model, loss_fn)` 会针对模型参数求梯度。
- `optimizer.update(model, grads)` 会根据梯度更新模型内部参数。
- MLX 是惰性计算，所以需要 `mx.eval(...)` 触发实际执行。

标准训练循环：

```python
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y_true)
optimizer.update(model, grads)
mx.eval(loss, model.parameters(), optimizer.state)
```

---

## PyTorch 和 MLX 的核心差异

PyTorch 更像是：参数自己记录梯度。

流程是：

```text
loss.backward()
↓
梯度存到参数的 .grad 里面
↓
optimizer.step() 读取 .grad 并更新参数
```

MLX 更像是：对函数求导，直接返回梯度。

流程是：

```text
nn.value_and_grad(...)
↓
返回 loss 和 grads
↓
optimizer.update(model, grads) 更新模型
```

---

## 这节课的核心结论

模型模块不是新的数学概念。

它只是把：

- 参数
- 前向计算
- 参数管理

封装成一个对象。

以后无论是 MLP、Transformer、GPT 还是 Diffusion，我们都会不断重复这个模式：

1. 定义模型
2. 输入数据
3. 得到预测
4. 计算 loss
5. 反向求梯度
6. optimizer 更新参数
