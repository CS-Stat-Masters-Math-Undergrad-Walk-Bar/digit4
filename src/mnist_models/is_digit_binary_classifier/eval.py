import torch
import torchvision
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent
DATA_ROOT = "/u/zup7mn/Classes/NN/digit4/src/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")

target_transform = lambda y: 1 if y < 10 else 0  # 1 = digit, 0 = not digit

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.1736, std=0.3317),
])

test_set = torchvision.datasets.EMNIST(
    root=DATA_ROOT,
    split="byclass",
    train=False,
    download=True,
    transform=test_transforms,
    target_transform=target_transform,
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
)


def build_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=1),
        torch.nn.Dropout(0.25),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        torch.nn.Dropout(0.25),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Dropout(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 6 * 6, 256),
        torch.nn.Dropout(0.25),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
    ).to(device)


def build_resnet():
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 1)
    return model.to(device)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall":    recall_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds),
        "confusion_matrix": cm,
    }


def print_results(name, metrics):
    print(f"=== {name} ===")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"                pred=not-digit  pred=digit")
    print(f"  true=not-digit    {cm[0,0]:8d}      {cm[0,1]:8d}")
    print(f"  true=digit        {cm[1,0]:8d}      {cm[1,1]:8d}")
    print()


models = [
    ("CNN        (best_cnn.pth)",    build_cnn,    REPO_ROOT / "state/best_cnn.pth"),
    ("ResNet-18  (best_resnet.pth)", build_resnet, REPO_ROOT / "best_resnet.pth"),
]

results = {}
for name, build_fn, weights_path in models:
    print(f"Loading {weights_path.name} ...")
    model = build_fn()
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    metrics = evaluate(model, test_loader)
    results[name] = metrics
    print_results(name, metrics)

winner = max(results, key=lambda n: results[n]["accuracy"])
print(f"Winner: {winner}  (accuracy {results[winner]['accuracy']:.4f})")
