import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def surprise(kl: torch.Tensor, lambda_s: float) -> torch.Tensor:
    """
    S(x, z) = 1 - exp(-lambda * KL)

    Normalizes KL from [0, inf) into [0, 1). lambda_s controls how quickly
    S saturates — calibrate so S ~ 0.5-0.8 at the warmup KL equilibrium.
    """
    kl_clamped = torch.clamp(kl, min=1e-8)
    return 1 - torch.exp(-lambda_s * kl_clamped)


def novelty(
    x_hat: torch.Tensor,
    digit_classifier: nn.Module,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,
    ambiguity_weight: float = 0.7,
    relevance_weight: float = 0.3,
) -> torch.Tensor:

    x_hat = x_hat.view(-1, 1, 28, 28)
    probs = F.softmax(digit_classifier(x_hat), dim=1)  # (B, 10)

    p_c1 = probs[:, c1]                                # (B,)
    p_c2 = probs[:, c2]                                # (B,)
    relevance = p_c1 + p_c2                            # (B,)

    # renormalize over {c1, c2} to isolate ambiguity between them
    p_tilde_c1 = p_c1 / (relevance + eps)
    p_tilde_c2 = p_c2 / (relevance + eps)

    # binary entropy normalized to [0, 1]
    h = -(
        p_tilde_c1 * torch.log(p_tilde_c1 + eps)
        + p_tilde_c2 * torch.log(p_tilde_c2 + eps)
    ) / torch.log(torch.tensor(2.0, device=x_hat.device))  # (B,)

    # geometric mean — non-compensatory, ambiguity-weighted
    n = h ** ambiguity_weight * relevance ** relevance_weight

    return n.mean()  # scalar in [0, 1]


def value(
    x_hat: torch.Tensor,
    digit_vs_nondigit_classifier: nn.Module,
    eps: float = 1e-8,
    emnist_mu: float = 0.1736,
    emnist_sd: float = 0.3317,
) -> torch.Tensor:
    """
    V(x) = P(digit | x) from the frozen EMNIST ResNet18 binary classifier.

    Normalization is applied inside this function because the EMNIST classifier
    was trained with mean=0.1736, std=0.3317, while the VAE decoder outputs
    raw [0, 1] pixel values.

    Label convention: 0 = digit, 1 = letter, so P(digit|x) = sigmoid(logit).
    """
    x_hat  = x_hat.view(-1, 1, 28, 28)
    x_norm = (x_hat - emnist_mu) / emnist_sd

    logits  = digit_vs_nondigit_classifier(x_norm)  # (B, 1)
    p_digit = torch.sigmoid(logits.squeeze(1))       # (B,)

    return p_digit.mean()  # scalar in [0, 1]


def log_creativity_score(
    x_hat: torch.Tensor,
    kl: torch.Tensor,
    digit_classifier: nn.Module,
    value_classifier: nn.Module,
    value_weight: float,
    novelty_weight: float,
    surprise_weight: float,
    lambda_s: float,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,
    log_means: Optional[dict] = None,
    log_stds: Optional[dict] = None,
) -> tuple:
    """
    Compute log DC = a1*log(V) + a2*log(N) + a3*log(S) — the training objective.

    When log_means and log_stds are provided (from creativity_diagnostic), each
    log component is z-score standardized before weighting. This ensures the weights
    a1, a2, a3 have consistent influence regardless of each component's raw variance.
    Z-scores are clamped to [-3, 3] to prevent gradient explosion.

    Returns:
        (log_dc, log_v, log_n, log_s, v, n, s)
        log_dc — training signal (standardized weighted sum)
        v, n, s — raw scores in [0, 1] for monitoring
    """
    v = value(x_hat, value_classifier, eps)
    n = novelty(x_hat, digit_classifier, c1, c2, eps)
    s = surprise(kl, lambda_s)

    log_v = torch.log(v + eps)
    log_n = torch.log(n + eps)
    log_s = torch.log(s + eps)

    if log_means is not None and log_stds is not None:
        log_v = torch.clamp((log_v - log_means["V"]) / (log_stds["V"] + eps), -3, 3)
        log_n = torch.clamp((log_n - log_means["N"]) / (log_stds["N"] + eps), -3, 3)
        log_s = torch.clamp((log_s - log_means["S"]) / (log_stds["S"] + eps), -3, 3)

    log_dc = value_weight * log_v + novelty_weight * log_n + surprise_weight * log_s

    return log_dc, log_v, log_n, log_s, v, n, s


## then we do DC = v^v * n^n * s^s, DC is our score

## Model Paths
## Discriminator:   state/GAN/checkpoints/discriminators/discriminator_last.pth
## Is_digit: state/mnist_models/is_digit_binary_classifier/cnn/best_cnn.pth
## Digit Classifier: state/mnist_models/digit_classifier/emnist_mixup_classifier.pth
## VAE (for getting kl from the encoding): state/VAE/full_VAE.pth


class DeepCreativity(nn.Module):
    def __init__(
        self,
        vae: nn.Module,
        digit_classifier: nn.Module,
        is_digit_classifier: nn.Module,
        discriminator: nn.Module,
        use_discriminator: bool = False,
        value_weight: float = 1.0,
        novelty_weight: float = 1.0,
        surprise_weight: float = 1.0,
        lambda_s: float = 1.0,
        c1: int = 2,
        c2: int = 6,
        eps: float = 1e-8,
        emnist_mu: float = 0.1736,
        emnist_sd: float = 0.3317,
        log_means: Optional[dict] = None,
        log_stds: Optional[dict] = None,
    ):
        super().__init__()
        self.vae = vae
        self.digit_classifier = digit_classifier
        self.is_digit_classifier = is_digit_classifier
        self.discriminator = discriminator
        self.use_discriminator = use_discriminator
        self.value_weight = value_weight
        self.novelty_weight = novelty_weight
        self.surprise_weight = surprise_weight
        self.lambda_s = lambda_s
        self.c1 = c1
        self.c2 = c2
        self.eps = eps
        self.emnist_mu = emnist_mu
        self.emnist_sd = emnist_sd
        self.log_means = log_means
        self.log_stds = log_stds

        for m in (vae, digit_classifier, is_digit_classifier, discriminator):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def kl(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        mu, log_var = self.vae.Encoder(x_flat)
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)
        if self.use_discriminator:
            p_digit = self.discriminator(x).view(-1)
        else:
            x_norm = (x - self.emnist_mu) / self.emnist_sd
            logits = self.is_digit_classifier(x_norm).view(-1)
            p_digit = torch.sigmoid(logits)
        return p_digit.clamp(self.eps, 1.0)

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)
        probs = F.softmax(self.digit_classifier(x), dim=1)
        p1, p2 = probs[:, self.c1], probs[:, self.c2]
        relevance = p1 + p2
        pt1 = p1 / (relevance + self.eps)
        pt2 = p2 / (relevance + self.eps)
        h = -(pt1 * torch.log(pt1 + self.eps) + pt2 * torch.log(pt2 + self.eps)) / torch.log(
            torch.tensor(2.0, device=x.device)
        )
        return (h * relevance).clamp(self.eps, 1.0)

    def surprise_score(self, kl: torch.Tensor) -> torch.Tensor:
        kl_c = kl.clamp(min=self.eps)
        return (1 - torch.exp(-self.lambda_s * kl_c)).clamp(self.eps, 1.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        kl_per = self.kl(x)
        v = self.value(x)
        n = self.novelty(x)
        s = self.surprise_score(kl_per)

        log_v = torch.log(v + self.eps)
        log_n = torch.log(n + self.eps)
        log_s = torch.log(s + self.eps)

        if self.log_means is not None and self.log_stds is not None:
            log_v = torch.clamp((log_v - self.log_means["V"]) / (self.log_stds["V"] + self.eps), -3, 3)
            log_n = torch.clamp((log_n - self.log_means["N"]) / (self.log_stds["N"] + self.eps), -3, 3)
            log_s = torch.clamp((log_s - self.log_means["S"]) / (self.log_stds["S"] + self.eps), -3, 3)

        log_dc = (
            self.value_weight * log_v
            + self.novelty_weight * log_n
            + self.surprise_weight * log_s
        )
        dc = torch.exp(log_dc)
        return {
            "dc": dc,
            "log_dc": log_dc,
            "value": v,
            "novelty": n,
            "surprise": s,
            "kl": kl_per,
        }

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["dc"]

