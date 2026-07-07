import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

print("=== MLX Linear Regression with nn.Module ===")


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 在 MLX 的 nn.Module 中，赋值为 mx.array 的成员会被作为参数管理
        self.w = mx.array(0.0)
        self.b = mx.array(0.0)

    def __call__(self, x):
        return self.w * x + self.b


x = mx.array([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

model = LinearRegressionModel()

lr = 0.01
steps = 200

optimizer = optim.SGD(learning_rate=lr)


def loss_fn(model, x, y_true):
    y_pred = model(x)
    loss = mx.mean((y_pred - y_true) ** 2)
    return loss


value_and_grad_fn = nn.value_and_grad(model, loss_fn)

for step in range(steps):
    loss, grads = value_and_grad_fn(model, x, y_true)

    optimizer.update(model, grads)

    mx.eval(loss, model.parameters(), optimizer.state)

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={float(loss):.6f} "
            f"w={float(model.w):.4f} "
            f"b={float(model.b):.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
print("\nModel parameters:")
print(model.parameters())
