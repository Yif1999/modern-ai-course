import torch

print("=== PyTorch MLP Regression Demo ===")

# 训练数据：学习 y = x^2
x = torch.linspace(-2.0, 2.0, 100).unsqueeze(1)
y_true = x ** 2


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

steps = 1000

for step in range(steps):
    y_pred = model(x)
    loss = ((y_pred - y_true) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == steps - 1:
        print(f"step={step:04d} loss={loss.item():.6f}")

print("\nSample predictions:")
test_x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
with torch.no_grad():
    test_y = model(test_x)

for x_value, y_value in zip(test_x.squeeze(), test_y.squeeze()):
    print(f"x={x_value.item(): .1f}, pred={y_value.item(): .4f}, true={(x_value ** 2).item(): .4f}")

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(name, tuple(param.shape))
