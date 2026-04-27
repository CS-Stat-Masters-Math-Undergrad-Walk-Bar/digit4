import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import ConcatDataset, DataLoader, Subset

import os
import wandb
from PIL import Image
from torchvision.utils import save_image


from src.loss import creative_ELBO, ELBO
from src.models.vae import VariationalAutoEncoder # neg_ELBO, neg_creative_ELBO
# from src.loss import neg_ELBO


# ── Device ─────────────────────────────────────────────────────────────────

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),  # scales to [0,1]
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            )
        ]
    )
    transform_test = transforms.ToTensor()  # scales to [0,1]

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_test
    )

    twos_dataset = Subset(train_dataset, (train_dataset.targets == 2).nonzero(as_tuple=True)[0])
    sixes_dataset = Subset(train_dataset, (train_dataset.targets == 6).nonzero(as_tuple=True)[0])

    test_twos = Subset(test_dataset, (test_dataset.targets == 2).nonzero(as_tuple=True)[0])
    test_sixes = Subset(test_dataset, (test_dataset.targets == 6).nonzero(as_tuple=True)[0])
    test_subset = ConcatDataset([test_twos, test_sixes])

    twos_sixes_loader = DataLoader(
        ConcatDataset([twos_dataset, sixes_dataset]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return twos_sixes_loader, test_loader


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
    gamma: float = 1.0, ## KL weighting term
    r_decay: float = 0.01,  # decay rate for reconstruction loss once creativity starts

):
    total_epochs = warmup_epochs + creative_epochs
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(total_epochs):
        vae.train()
        is_warmup = epoch < warmup_epochs
        phase = "warmup" if is_warmup else "creative"

        train_loss = 0.0
        sum_recon = sum_kl = sum_v = sum_n = sum_s = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            optimizer.zero_grad()

            pred = vae(x_flat)  # returns (x_hat, mu, log_var)

            if is_warmup:
                reconstruction, kl = ELBO(pred, x_flat)
                loss = reconstruction + kl * gamma # always train on ELBO during warmup
            else:


                reconstruction, kl, v, n, s = creative_ELBO(
                    pred,
                    x_flat,
                    # decoder_dist,
                    digit_classifier,
                    value_classifier,
                    value_weight,
                    novelty_weight,
                    surprise_weight,
                    lambda_s,
                    c1,
                    c2,
                )

                loss = reconstruction * r_decay+ kl - (value_weight * v + novelty_weight * n + surprise_weight * s)
                sum_v += v.item(); sum_n += n.item(); sum_s += s.item()

            sum_recon += reconstruction.item(); sum_kl += kl.item()

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
                reconstruction, kl = ELBO(pred, x_flat)  
                loss = reconstruction + kl * gamma # always eval on ELBO
                test_loss += loss.item()


            # ── sample generation ───────────────────────────────────────────
            # Get 6 samples from the test loader
            sample_x, _ = next(iter(test_loader))
            sample_x = sample_x[:6].to(device)
            sample_x_flat = sample_x.view(sample_x.size(0), -1)
            
            # Reconstruct the samples
            pred_sample = vae(sample_x_flat)
            recon_x_flat = pred_sample[0]
            
            # Reshape back to image dimensions (assuming 28x28 for MNIST)
            recon_x = recon_x_flat.view(6, 1, 28, 28)
            
            # Combine original and reconstructed images
            # Top row: original, Bottom row: reconstructed
            comparison = torch.cat([sample_x, recon_x])
            
            # Save the grid
            sample_dir = "/u/zup7mn/Classes/NN/digit4/src/creative_vae/samples"
            os.makedirs(sample_dir, exist_ok=True)
            save_image(
                comparison.cpu(), 
                f"{sample_dir}/reconstruction_epoch_{epoch + 1}.png", 
                nrow=6
            )


        N = len(train_loader)
        log = {"epoch": epoch + 1, "train_loss": train_loss/N, "test_loss": test_loss/len(test_loader), "recon": sum_recon/N, "kl": sum_kl/N}
        if not is_warmup:
            log |= {"v": sum_v/N, "n": sum_n/N, "s": sum_s/N}
        wandb.log(log)

        creative_str = f" v={sum_v/N:.4f} n={sum_n/N:.4f} s={sum_s/N:.4f}" if not is_warmup else ""
        print(
            f"Epoch {epoch + 1}/{total_epochs} [{phase}] "
            f"train={train_loss/N:.4f} test={test_loss/len(test_loader):.4f} "
            f"recon={sum_recon/N:.4f} kl={sum_kl/N:.4f}{creative_str}"
        )

        torch.save(
            vae.state_dict(), f"src/creative_vae/checkpoints/vae_epoch{epoch + 1}.pth"
        )

    # ── Create GIF from saved samples ──────────────────────────────────────────
    print("Creating training animation GIF...")
    sample_dir = "/u/zup7mn/Classes/NN/digit4/src/creative_vae/samples"
    frames = []
    for i in range(total_epochs):
        img_path = f"{sample_dir}/reconstruction_epoch_{i + 1}.png"
        if os.path.exists(img_path):
            frames.append(Image.open(img_path))
    
    if frames:
        frames[0].save(
            f"{sample_dir}/training_progress.gif",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved GIF to {sample_dir}/training_progress.gif")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    wandb.init(project="creative-vae")
    os.makedirs("checkpoints", exist_ok=True)

    digit_clf, value_clf = load_classifiers(
        value_path="src/mnist_classifier/best_model.pth",
        digit_path="state/mnist_classifier/mnist_mixup_classifier.pth",
        device=device,
    )

    train_loader, test_loader = get_dataloaders(batch_size=256)

    vae = VariationalAutoEncoder(
        real_dim=784, h_dims=[512, 256], bn_dim=16, # decoder_type="gaussian"
    ).to(device)

    train(
        vae=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        digit_classifier=digit_clf,
        value_classifier=value_clf,
        warmup_epochs=30,  # train standard ELBO first
        creative_epochs=60,  # then introduce creativity terms
        r_decay=0.5,  # decay rate for reconstruction loss once creativity starts
        # decoder_dist="gaussian",
        value_weight=10.00,
        novelty_weight=1.00,
        surprise_weight=0.33,
        lambda_s=1.0,
        c1=2,
        c2=6,
        lr=1e-4,
        device=device,
    )
