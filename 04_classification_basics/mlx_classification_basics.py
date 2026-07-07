import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

print("=== MLX Classification Basics Demo ===")

mx.random.seed(42)

# 构造二维输入数据
num_samples = 200
x = mx.random.normal((num_samples, 2))

# 分类规则：如果 x1 + x2 > 0，则类别为 1，否则为 0
y = (x[:, 0] + x[:, 1] > 0).astype(mx.int32)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 16)
        self.linear2 = nn.Linear(16, 2)

    def __call__(self, x):
        h = self.linear1(x)
        h = nn.relu(h)
        logits = self.linear2(h)
        return logits


model = Classifier()
optimizer = optim.Adam(learning_rate=0.01)

steps = 500


def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    return loss


value_and_grad_fn = nn.value_and_grad(model, loss_fn)

for step in range(steps):
    loss, grads = value_and_grad_fn(model, x, y)

    optimizer.update(model, grads)

    logits = model(x)
    predictions = mx.argmax(logits, axis=1)
    accuracy = mx.mean((predictions == y).astype(mx.float32))

    mx.eval(loss, accuracy, model.parameters(), optimizer.state)

    if step % 50 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={float(loss):.6f} "
            f"accuracy={float(accuracy):.4f}"
        )

print("\nSample predictions:")
test_x = mx.array([
    [2.0, 1.0],
    [-2.0, -1.0],
    [1.0, -2.0],
    [-1.0, 2.0],
])

test_logits = model(test_x)
test_probs = nn.softmax(test_logits, axis=1)
test_preds = mx.argmax(test_logits, axis=1)
mx.eval(test_logits, test_probs, test_preds)

for i in range(test_x.shape[0]):
    x1 = float(test_x[i, 0])
    x2 = float(test_x[i, 1])
    true_label = int(x1 + x2 > 0)
    pred_label = int(test_preds[i])
    probs = [round(float(test_probs[i, j]), 4) for j in range(2)]
    print(
        f"x=({x1: .1f}, {x2: .1f}) "
        f"true={true_label} "
        f"pred={pred_label} "
        f"probs={probs}"
    )

print("\nLogits shape:", test_logits.shape)
print("Each row has 2 numbers because this is a 2-class classifier.")
