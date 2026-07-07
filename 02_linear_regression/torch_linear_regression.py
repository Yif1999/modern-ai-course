import torch

print("=== PyTorch Linear Regression Demo ===")

# 训练数据：真实规律是 y = 3x + 2
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

# 待训练参数：一开始故意设错
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01
steps = 200

for step in range(steps):
    # forward
    y_pred = w * x + b

    # mean squared error
    loss = ((y_pred - y_true) ** 2).mean()

    # backward
    loss.backward()

    # manual gradient descent
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # clear gradients
    w.grad.zero_()
    b.grad.zero_()

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={loss.item():.6f} "
            f"w={w.item():.4f} "
            f"b={b.item():.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
