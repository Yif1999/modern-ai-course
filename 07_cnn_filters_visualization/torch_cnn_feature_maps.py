from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


print("=== CNN Filters / Feature Maps Visualization ===")

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
print("Data dir:", data_dir)
print("Output dir:", output_dir)

batch_size = 128
epochs = 1
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(
    root=str(data_dir),
    train=True,
    download=True,
    transform=transform,
)

test_dataset = torchvision.datasets.MNIST(
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


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.classifier = torch.nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        logits = self.classifier(x)
        return logits

    def forward_features(self, x):
        conv1_out = self.conv1(x)
        relu1_out = torch.relu(conv1_out)
        pool1_out = self.pool(relu1_out)

        conv2_out = self.conv2(pool1_out)
        relu2_out = torch.relu(conv2_out)
        pool2_out = self.pool(relu2_out)

        return {
            "conv1_out": conv1_out,
            "relu1_out": relu1_out,
            "pool1_out": pool1_out,
            "conv2_out": conv2_out,
            "relu2_out": relu2_out,
            "pool2_out": pool2_out,
        }


model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\nParameter shapes:")
print("conv1.weight:", tuple(model.conv1.weight.shape))
print("conv1.bias:", tuple(model.conv1.bias.shape))
print("conv2.weight:", tuple(model.conv2.weight.shape))
print("conv2.bias:", tuple(model.conv2.bias.shape))
print("classifier.weight:", tuple(model.classifier.weight.shape))
print("classifier.bias:", tuple(model.classifier.bias.shape))


def evaluate(model, data_loader):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            total += labels.shape[0]
            correct += (predictions == labels).sum().item()

    model.train()
    return correct / total


print("\nTraining for 1 epoch...")

for epoch in range(epochs):
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total += labels.shape[0]
        correct += (predictions == labels).sum().item()

        if batch_idx % 150 == 0:
            print(
                f"epoch={epoch + 1} "
                f"batch={batch_idx:03d} "
                f"loss={loss.item():.6f}"
            )

    train_acc = correct / total
    test_acc = evaluate(model, test_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train_acc={train_acc:.4f} "
        f"test_acc={test_acc:.4f}"
    )

print("\nVisualizing conv1 filters...")

conv1_weights = model.conv1.weight.detach().cpu()
# shape: [8, 1, 3, 3]

fig, axes = plt.subplots(2, 4, figsize=(8, 4))

for i, ax in enumerate(axes.flat):
    kernel = conv1_weights[i, 0]
    ax.imshow(kernel, cmap="gray")
    ax.set_title(f"filter {i}")
    ax.axis("off")

fig.suptitle("Conv1 learned 3x3 filters")
fig.tight_layout()

filters_path = output_dir / "conv1_filters.png"
fig.savefig(filters_path, dpi=150)
plt.close(fig)

print("Saved:", filters_path)

print("\nVisualizing feature maps for one test image...")

model.eval()

sample_image, sample_label = test_dataset[0]
sample_batch = sample_image.unsqueeze(0).to(device)

with torch.no_grad():
    features = model.forward_features(sample_batch)
    logits = model(sample_batch)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1).item()
    confidence = probs[0, pred].item()

print("Sample true label:", sample_label)
print("Sample predicted label:", pred)
print("Confidence:", round(confidence, 4))

conv1_feature_maps = features["relu1_out"].detach().cpu()[0]
# shape: [8, 28, 28]

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flat

axes[0].imshow(sample_image.squeeze(0), cmap="gray")
axes[0].set_title(f"input\nlabel={sample_label}")
axes[0].axis("off")

for i in range(8):
    axes[i + 1].imshow(conv1_feature_maps[i], cmap="gray")
    axes[i + 1].set_title(f"map {i}")
    axes[i + 1].axis("off")

axes[9].axis("off")

fig.suptitle("Input image and conv1 feature maps")
fig.tight_layout()

feature_maps_path = output_dir / "conv1_feature_maps.png"
fig.savefig(feature_maps_path, dpi=150)
plt.close(fig)

print("Saved:", feature_maps_path)

print("\nFeature shapes for the sample image:")

with torch.no_grad():
    for name, value in features.items():
        print(name, tuple(value.shape))

print("\nDone.")
