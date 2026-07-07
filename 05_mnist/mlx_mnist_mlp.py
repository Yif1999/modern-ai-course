from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

print("=== MLX MNIST MLP Demo ===")
print("MLX default device:", mx.default_device())

data_dir = Path(__file__).resolve().parent / "data"
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


def dataset_to_numpy(dataset):
    images = []
    labels = []

    for image, label in dataset:
        # image shape: [1, 28, 28]
        image_np = image.numpy().reshape(-1)  # [784]
        images.append(image_np)
        labels.append(label)

    images = np.stack(images).astype(np.float32)
    labels = np.array(labels).astype(np.int32)
    return images, labels


print("Converting dataset to numpy arrays...")
train_images_np, train_labels_np = dataset_to_numpy(train_dataset)
test_images_np, test_labels_np = dataset_to_numpy(test_dataset)

print("train_images_np shape:", train_images_np.shape)
print("train_labels_np shape:", train_labels_np.shape)
print("test_images_np shape:", test_images_np.shape)
print("test_labels_np shape:", test_labels_np.shape)

train_images = mx.array(train_images_np)
train_labels = mx.array(train_labels_np)
test_images = mx.array(test_images_np)
test_labels = mx.array(test_labels_np)
mx.eval(train_images, train_labels, test_images, test_labels)


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.linear2 = nn.Linear(128, 10)

    def __call__(self, x):
        h = self.linear1(x)
        h = nn.relu(h)
        logits = self.linear2(h)
        return logits


model = MNISTMLP()
optimizer = optim.Adam(learning_rate=learning_rate)


def loss_fn(model, images, labels):
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    return loss


value_and_grad_fn = nn.value_and_grad(model, loss_fn)


def batch_iter(images, labels, batch_size, shuffle=True):
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        batch_indices = mx.array(indices[start:start + batch_size])
        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_images, batch_labels


def evaluate(model, images, labels, batch_size):
    total = 0
    correct = 0
    total_loss = 0.0

    for batch_images, batch_labels in batch_iter(
        images,
        labels,
        batch_size,
        shuffle=False,
    ):
        logits = model(batch_images)
        loss = nn.losses.cross_entropy(logits, batch_labels, reduction="mean")
        predictions = mx.argmax(logits, axis=1)
        correct_batch = mx.sum((predictions == batch_labels).astype(mx.int32))
        mx.eval(loss, correct_batch)

        total += batch_labels.shape[0]
        correct += int(correct_batch)
        total_loss += float(loss) * batch_labels.shape[0]

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


for epoch in range(epochs):
    total = 0
    correct = 0
    total_loss = 0.0

    for batch_idx, (batch_images, batch_labels) in enumerate(
        batch_iter(train_images, train_labels, batch_size, shuffle=True)
    ):
        loss, grads = value_and_grad_fn(model, batch_images, batch_labels)
        optimizer.update(model, grads)

        logits = model(batch_images)
        predictions = mx.argmax(logits, axis=1)
        correct_batch = mx.sum((predictions == batch_labels).astype(mx.int32))
        mx.eval(loss, correct_batch, model.parameters(), optimizer.state)

        total += batch_labels.shape[0]
        correct += int(correct_batch)
        total_loss += float(loss) * batch_labels.shape[0]

        if batch_idx % 100 == 0:
            print(
                f"epoch={epoch + 1} "
                f"batch={batch_idx:03d} "
                f"loss={float(loss):.6f}"
            )

    train_loss = total_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(model, test_images, test_labels, batch_size)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"train_loss={train_loss:.6f} "
        f"train_acc={train_acc:.4f} "
        f"test_loss={test_loss:.6f} "
        f"test_acc={test_acc:.4f}"
    )

print("\nSample predictions:")
sample_images = test_images[:10]
sample_labels = test_labels[:10]
sample_logits = model(sample_images)
sample_probs = nn.softmax(sample_logits, axis=1)
sample_predictions = mx.argmax(sample_logits, axis=1)
mx.eval(sample_probs, sample_predictions)

for i in range(10):
    true_label = int(sample_labels[i])
    pred_label = int(sample_predictions[i])
    confidence = float(sample_probs[i, pred_label])
    print(
        f"sample={i} "
        f"true={true_label} "
        f"pred={pred_label} "
        f"confidence={confidence:.4f}"
    )

print("\nDone.")
