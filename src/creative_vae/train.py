import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from src.loss import neg_ELBO, neg_creative_ELBO
from src.models.vae import VariationalAutoEncoder


# ── Device ─────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# ── Classifier definitions ──────────────────────────────────────────────────


def build_value_classifier() -> nn.Module:
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 1)
    return model


def build_digit_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(64 * 676, 10),
    )


def load_classifiers(
    value_path: str,
    digit_path: str,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """
    Load, freeze, and return (digit_classifier, value_classifier).
    """
    # ── value classifier (EMNIST ResNet18 binary) ───────────────────────
    value_clf = build_value_classifier()
    value_clf.load_state_dict(torch.load(value_path, map_location=device))
    value_clf.to(device)
    value_clf.eval()
    for p in value_clf.parameters():
        p.requires_grad_(False)

    # ── digit classifier (MNIST 10-class MixUp CNN) ──────────────────────
    digit_clf = build_digit_classifier()
    digit_clf.load_state_dict(torch.load(digit_path, map_location=device))
    digit_clf.to(device)
    digit_clf.eval()
    for p in digit_clf.parameters():
        p.requires_grad_(False)

    return digit_clf, value_clf


# ── Data ────────────────────────────────────────────────────────────────────


def get_dataloaders(batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
    """
    MNIST train and test loaders.
    No normalization — MNIST digit classifier was trained on [0,1].
    EMNIST normalization is applied inside value() in metrics.py.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # scales to [0,1]
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


# ── Training loop ───────────────────────────────────────────────────────────


def train(
    vae: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    digit_classifier: nn.Module,
    value_classifier: nn.Module,
    warmup_epochs: int,
    creative_epochs: int,
    decoder_dist: str = "gaussian",
    value_weight: float = 0.33,
    novelty_weight: float = 0.33,
    surprise_weight: float = 0.33,
    lambda_s: float = 1.0,  # surprise saturation rate
    c1: int = 2,  # target class 1
    c2: int = 6,  # target class 2
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    total_epochs = warmup_epochs + creative_epochs
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(total_epochs):
        vae.train()
        is_warmup = epoch < warmup_epochs
        phase = "warmup" if is_warmup else "creative"

        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            optimizer.zero_grad()

            pred = vae(x_flat)  # returns (x_hat, mu, log_var)

            if is_warmup:
                loss = neg_ELBO(pred, x_flat, decoder_dist)
            else:
                loss = neg_creative_ELBO(
                    pred,
                    x_flat,
                    decoder_dist,
                    digit_classifier,
                    value_classifier,
                    value_weight,
                    novelty_weight,
                    surprise_weight,
                    lambda_s,
                    c1,
                    c2,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── evaluation ──────────────────────────────────────────────────
        vae.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_flat = x.view(x.size(0), -1)
                pred = vae(x_flat)
                loss = neg_ELBO(pred, x_flat, decoder_dist)  # always eval on ELBO
                test_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{total_epochs} [{phase}] "
            f"train={train_loss / len(train_loader):.4f} "
            f"test={test_loss / len(test_loader):.4f}"
        )

        torch.save(
            vae.state_dict(), f"/src/creative_vae/checkpoints/vae_epoch{epoch + 1}.pth"
        )


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    os.makedirs("checkpoints", exist_ok=True)

    digit_clf, value_clf = load_classifiers(
        value_path="src/mnist_classifier/best_model.pth",
        digit_path="src/mnist_classifier/mnist_mixup_classifier.pth",
        device=device,
    )

    train_loader, test_loader = get_dataloaders(batch_size=256)

    vae = VariationalAutoEncoder(
        real_dim=784, h_dims=[512, 256], bn_dim=16, decoder_type="gaussian"
    ).to(device)

    train(
        vae=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        digit_classifier=digit_clf,
        value_classifier=value_clf,
        warmup_epochs=20,  # train standard ELBO first
        creative_epochs=30,  # then introduce creativity terms
        decoder_dist="gaussian",
        value_weight=0.33,
        novelty_weight=0.33,
        surprise_weight=0.33,
        lambda_s=1.0,
        c1=2,
        c2=6,
        lr=1e-3,
        device=device,
    )
