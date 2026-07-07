# Tensor / Array / Autograd Notes

## PyTorch

- `requires_grad=True` 表示这个张量需要被求梯度。
- `y.backward()` 会从 y 开始反向传播。
- 梯度保存在 `w.grad` 中。

## MLX

- MLX 不使用 `requires_grad=True` 和 `backward()` 这种写法。
- MLX 更接近函数式自动求导。
- `mx.value_and_grad(f)(w)` 会同时返回函数值和梯度。
- MLX 是 lazy evaluation，所以通常需要 `mx.eval(...)` 触发实际计算。

## 共同点

对于：

```text
y = x1*w1 + x2*w2 + x3*w3
```

有：

```text
dy/dw = x
```

所以当：

```text
x = [1, 2, 3]
```

梯度就是：

```text
[1, 2, 3]
```
