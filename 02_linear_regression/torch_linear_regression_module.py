import torch

print("=== PyTorch Linear Regression with nn.Module ===")


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Parameter 表示这是模型中需要训练的参数
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.w * x + self.b


x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

model = LinearRegressionModel()

lr = 0.01
steps = 200

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for step in range(steps):
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={loss.item():.6f} "
            f"w={model.w.item():.4f} "
            f"b={model.b.item():.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(name, param.data)
