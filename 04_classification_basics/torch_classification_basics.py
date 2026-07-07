import torch

print("=== PyTorch Classification Basics Demo ===")

torch.manual_seed(42)

# 构造二维输入数据
num_samples = 200
x = torch.randn(num_samples, 2)

# 分类规则：如果 x1 + x2 > 0，则类别为 1，否则为 0
y = (x[:, 0] + x[:, 1] > 0).long()


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

steps = 500

for step in range(steps):
    logits = model(x)

    # CrossEntropyLoss 接收 logits，不需要我们手动 softmax
    loss = torch.nn.functional.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0 or step == steps - 1:
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()
        print(
            f"step={step:03d} "
            f"loss={loss.item():.6f} "
            f"accuracy={accuracy.item():.4f}"
        )

print("\nSample predictions:")
test_x = torch.tensor([
    [2.0, 1.0],
    [-2.0, -1.0],
    [1.0, -2.0],
    [-1.0, 2.0],
])

with torch.no_grad():
    test_logits = model(test_x)
    test_probs = torch.softmax(test_logits, dim=1)
    test_preds = torch.argmax(test_logits, dim=1)

for i in range(test_x.shape[0]):
    x1 = test_x[i, 0].item()
    x2 = test_x[i, 1].item()
    true_label = int(x1 + x2 > 0)
    pred_label = test_preds[i].item()
    probs = test_probs[i].tolist()
    print(
        f"x=({x1: .1f}, {x2: .1f}) "
        f"true={true_label} "
        f"pred={pred_label} "
        f"probs={[round(p, 4) for p in probs]}"
    )

print("\nLogits shape:", tuple(test_logits.shape))
print("Each row has 2 numbers because this is a 2-class classifier.")
