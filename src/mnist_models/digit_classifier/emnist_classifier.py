# %%
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.transforms import v2

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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
mnist_val_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# %%
DATA_ROOT = "/u/zup7mn/Classes/NN/digit4/src/data"

emnist_train = torchvision.datasets.EMNIST(
    root=DATA_ROOT, split="digits", train=True, download=True, transform=emnist_train_transforms
)
emnist_test = torchvision.datasets.EMNIST(
    root=DATA_ROOT, split="digits", train=False, download=True, transform=emnist_val_transforms
)
mnist_test = torchvision.datasets.MNIST(
    root=DATA_ROOT, train=False, download=True, transform=mnist_val_transforms
)

# %%
BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True)
emnist_test_loader = torch.utils.data.DataLoader(emnist_test, batch_size=BATCH_SIZE)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE)

# %%
model = nn.Sequential(
    nn.Conv2d(1, 256, kernel_size=5, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    nn.Dropout(0.2),
    nn.Flatten(),
    nn.Linear(128 * 676, 10),
).to(device)

# %%
model.compile()

# %%
EPOCHS = 20
USE_MIXUP = True
mixup = v2.MixUp(num_classes=10)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss(reduction="mean")

if USE_MIXUP:
    print("Using mixup augmentation")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        if USE_MIXUP:
            images, labels = mixup(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in emnist_test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    print(f"  EMNIST val accuracy: {correct / total:.4f}")

    if USE_MIXUP:
        torch.save(model.state_dict(), "emnist_mixup_classifier.pth")
    else:
        torch.save(model.state_dict(), "emnist_classifier.pth")

# %%
def compute_accuracy(mdl, loader):
    mdl.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (mdl(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

emnist_acc = compute_accuracy(model, emnist_test_loader)
mnist_acc = compute_accuracy(model, mnist_test_loader)
print(f"Final EMNIST test accuracy: {emnist_acc:.4f}")
print(f"Final MNIST  test accuracy: {mnist_acc:.4f}")
