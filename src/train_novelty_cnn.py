# %%
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.transforms import v2

from project_paths import DATA_ROOT, NOVELTY_CNN_MIXUP, NOVELTY_CNN_PLAIN

BATCH_SIZE = 256

# %%
# EMNIST images are stored transposed relative to MNIST, so we flip H/W after tensorizing
emnist_train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x.permute(0, 2, 1)),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),
    v2.CenterCrop((28, 28)),
])
emnist_val_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x.permute(0, 2, 1)),
])


# %%
def get_device():
    return torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"


def build_model(device):
    return nn.Sequential(
        nn.Conv2d(1, 256, kernel_size=5, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(128 * 676, 10),
    ).to(device)


def get_loaders():
    train_set = torchvision.datasets.EMNIST(
        root=DATA_ROOT, split="digits", train=True, download=True, transform=emnist_train_transforms
    )
    test_set = torchvision.datasets.EMNIST(
        root=DATA_ROOT, split="digits", train=False, download=True, transform=emnist_val_transforms
    )
    return (
        torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE),
    )


# %%
def train(epochs=20, use_mixup=True, device=None):
    if device is None:
        device = get_device()

    model = build_model(device)
    model.compile()
    train_loader, test_loader = get_loaders()

    mixup = v2.MixUp(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_mixup:
                images, labels = mixup(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[EMNIST] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        print(f"  EMNIST test accuracy: {correct / total:.4f}")

        torch.save(
            model.state_dict(),
            NOVELTY_CNN_MIXUP if use_mixup else NOVELTY_CNN_PLAIN,
        )

    return model, test_loader


# %%
if __name__ == "__main__":
    device = get_device()
    print(f"Using {device} device")
    train(epochs=20, use_mixup=True, device=device)
