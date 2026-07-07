import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

print("=== MLX MLP Regression Demo ===")

# 训练数据：学习 y = x^2
x = mx.linspace(-2.0, 2.0, 100).reshape(100, 1)
y_true = x ** 2


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 1)

    def __call__(self, x):
        h = self.linear1(x)
        h = nn.relu(h)
        y = self.linear2(h)
        return y


model = MLP()
optimizer = optim.Adam(learning_rate=0.01)

steps = 1000


def loss_fn(model, x, y_true):
    y_pred = model(x)
    loss = mx.mean((y_pred - y_true) ** 2)
    return loss


value_and_grad_fn = nn.value_and_grad(model, loss_fn)

for step in range(steps):
    loss, grads = value_and_grad_fn(model, x, y_true)

    optimizer.update(model, grads)

    mx.eval(loss, model.parameters(), optimizer.state)

    if step % 100 == 0 or step == steps - 1:
        print(f"step={step:04d} loss={float(loss):.6f}")

print("\nSample predictions:")
test_x = mx.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
test_y = model(test_x)
mx.eval(test_y)

for i in range(test_x.shape[0]):
    x_value = float(test_x[i, 0])
    pred_value = float(test_y[i, 0])
    true_value = x_value ** 2
    print(f"x={x_value: .1f}, pred={pred_value: .4f}, true={true_value: .4f}")

print("\nModel parameters:")
print(model.parameters())
