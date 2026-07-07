from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms


print("=== PyTorch CNN MNIST Demo ===")

data_dir = Path(__file__).resolve().parent / "data"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
print("Data directory:", data_dir)

batch_size = 128
epochs = 2
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

images, labels = next(iter(train_loader))

print("\nInput shape before model:")
print("images shape:", tuple(images.shape))
print("labels shape:", tuple(labels.shape))

flattened = images.view(images.shape[0], -1)

print("\nAfter flatten:")
print("flattened shape:", tuple(flattened.shape))

print("\nMeaning:")
print("[batch, 1, 28, 28] keeps image structure.")
print("[batch, 784] loses 2D spatial structure.")


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
        # x: [batch, 1, 28, 28]
        x = self.conv1(x)
        # [batch, 8, 28, 28]
        x = torch.relu(x)

        x = self.pool(x)
        # [batch, 8, 14, 14]

        x = self.conv2(x)
        # [batch, 16, 14, 14]
        x = torch.relu(x)

        x = self.pool(x)
        # [batch, 16, 7, 7]

        x = x.view(x.shape[0], -1)
        # [batch, 16 * 7 * 7]

        logits = self.classifier(x)
        # [batch, 10]

        return logits


model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(model, data_loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total += labels.shape[0]
            correct += (predictions == labels).sum().item()
            total_loss += loss.item() * labels.shape[0]

    model.train()
    return total_loss / total, correct / total


print("\nChecking feature shapes with one batch:")

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

for epoch in range(epochs):
    total = 0
    correct = 0
    total_loss = 0.0

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
        total_loss += loss.item() * labels.shape[0]

        if batch_idx % 150 == 0:
            print(
                f"epoch={epoch + 1} "
                f"batch={batch_idx:03d} "
                f"loss={loss.item():.6f}"
            )

    train_loss = total_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train_loss={train_loss:.6f} "
        f"train_acc={train_acc:.4f} "
        f"test_loss={test_loss:.6f} "
        f"test_acc={test_acc:.4f}"
    )

print("\nSample predictions:")
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images[:10])
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)

for i in range(10):
    true_label = labels[i].item()
    pred_label = predictions[i].item()
    confidence = probs[i, pred_label].item()
    print(
        f"sample={i} "
        f"true={true_label} "
        f"pred={pred_label} "
        f"confidence={confidence:.4f}"
    )

print("\nDone.")
