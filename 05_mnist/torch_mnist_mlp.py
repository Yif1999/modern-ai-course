from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

print("=== PyTorch MNIST MLP Demo ===")

data_dir = Path(__file__).resolve().parent / "data"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

batch_size = 128
epochs = 3
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


class MNISTMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = x.view(x.shape[0], -1)
        # after flatten: [batch, 784]
        logits = self.net(x)
        return logits


model = MNISTMLP().to(device)
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

    avg_loss = total_loss / total
    accuracy = correct / total
    model.train()
    return avg_loss, accuracy


for epoch in range(epochs):
    model.train()
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

        if batch_idx % 100 == 0:
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
