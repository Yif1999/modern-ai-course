import torch

print("=== PyTorch Linear Regression with Optimizer ===")

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01
steps = 200

optimizer = torch.optim.SGD([w, b], lr=lr)

for step in range(steps):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={loss.item():.6f} "
            f"w={w.item():.4f} "
            f"b={b.item():.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
