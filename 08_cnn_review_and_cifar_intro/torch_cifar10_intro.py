from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


print("=== CNN Review: CIFAR-10 Intro ===")

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print("Data dir:", data_dir)
print("Output dir:", output_dir)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

batch_size = 128
epochs = 10
learning_rate = 0.001

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    download=True,
    transform=transform,
)

test_dataset = torchvision.datasets.CIFAR10(
    root=str(data_dir),
    train=False,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

images, labels = next(iter(train_loader))

print("\nCIFAR-10 batch shape:")
print("images:", tuple(images.shape))
print("labels:", tuple(labels.shape))

print("\nMeaning:")
print("[batch, 3, 32, 32]")
print("batch: 一次处理多少张图片")
print("3: RGB 三个颜色通道")
print("32, 32: 图片高和宽")

print("\nFirst 16 labels:")
print([class_names[int(label)] for label in labels[:16]])

grid = torchvision.utils.make_grid(images[:16], nrow=8, padding=2)
grid_np = grid.permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 5))
plt.imshow(grid_np)
plt.axis("off")
plt.title("CIFAR-10 samples")

sample_path = output_dir / "cifar10_samples.png"
plt.savefig(sample_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved sample image:", sample_path)


class SmallCIFARCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.classifier = torch.nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        # x: [batch, 3, 32, 32]
        x = self.conv1(x)
        # [batch, 16, 32, 32]
        x = torch.relu(x)

        x = self.pool(x)
        # [batch, 16, 16, 16]

        x = self.conv2(x)
        # [batch, 32, 16, 16]
        x = torch.relu(x)

        x = self.pool(x)
        # [batch, 32, 8, 8]

        x = x.view(x.shape[0], -1)
        # [batch, 32 * 8 * 8]

        logits = self.classifier(x)
        # [batch, 10]

        return logits


model = SmallCIFARCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\nModel parameter shapes:")
print("conv1.weight:", tuple(model.conv1.weight.shape))
print("conv1.bias:", tuple(model.conv1.bias.shape))
print("conv2.weight:", tuple(model.conv2.weight.shape))
print("conv2.bias:", tuple(model.conv2.bias.shape))
print("classifier.weight:", tuple(model.classifier.weight.shape))
print("classifier.bias:", tuple(model.classifier.bias.shape))

print("\nChecking feature shapes with one mini-batch:")

model.eval()
with torch.no_grad():
    sample = images[:4].to(device)

    h1 = model.conv1(sample)
    h1_relu = torch.relu(h1)
    h1_pool = model.pool(h1_relu)

    h2 = model.conv2(h1_pool)
    h2_relu = torch.relu(h2)
    h2_pool = model.pool(h2_relu)

    print("sample:", tuple(sample.shape))
    print("after conv1:", tuple(h1.shape))
    print("after pool1:", tuple(h1_pool.shape))
    print("after conv2:", tuple(h2.shape))
    print("after pool2:", tuple(h2_pool.shape))
    print("after flatten:", tuple(h2_pool.view(h2_pool.shape[0], -1).shape))

model.train()


def evaluate(model, data_loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_images)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            predictions = torch.argmax(logits, dim=1)

            total += batch_labels.shape[0]
            correct += (predictions == batch_labels).sum().item()
            total_loss += loss.item() * batch_labels.shape[0]

    model.train()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def synchronize_device():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


print(f"\nTraining small CNN on CIFAR-10 for {epochs} epochs...")

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
epoch_times = []

synchronize_device()
training_start = time.perf_counter()

for epoch in range(epochs):
    synchronize_device()
    epoch_start = time.perf_counter()

    total = 0
    correct = 0
    total_loss = 0.0

    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_images)
        loss = torch.nn.functional.cross_entropy(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        total += batch_labels.shape[0]
        correct += (predictions == batch_labels).sum().item()
        total_loss += loss.item() * batch_labels.shape[0]

        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch + 1} "
                f"batch={batch_idx:03d} "
                f"loss={loss.item():.6f}"
            )

    train_loss = total_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_loader)

    synchronize_device()
    epoch_time = time.perf_counter() - epoch_start

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    epoch_times.append(epoch_time)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train_loss={train_loss:.6f} "
        f"train_acc={train_acc:.4f} "
        f"test_loss={test_loss:.6f} "
        f"test_acc={test_acc:.4f} "
        f"time={epoch_time:.2f}s"
    )

synchronize_device()
total_training_time = time.perf_counter() - training_start
print(f"\nTotal training time: {total_training_time:.2f}s")
print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")

epochs_axis = list(range(1, epochs + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, train_losses, marker="o", label="train loss")
plt.plot(epochs_axis, test_losses, marker="o", label="test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CIFAR-10 Small CNN Loss Curve")
plt.xticks(epochs_axis)
plt.grid(True, alpha=0.3)
plt.legend()

loss_curve_path = output_dir / "loss_curve.png"
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved loss curve:", loss_curve_path)

print("\nSample predictions:")

model.eval()
test_images, test_labels = next(iter(test_loader))
test_images = test_images.to(device)
test_labels = test_labels.to(device)

with torch.no_grad():
    logits = model(test_images[:10])
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)

for i in range(10):
    true_label = int(test_labels[i])
    pred_label = int(predictions[i])
    confidence = float(probs[i, pred_label])

    print(
        f"sample={i} "
        f"true={class_names[true_label]} "
        f"pred={class_names[pred_label]} "
        f"confidence={confidence:.4f}"
    )

print("\nDone.")
