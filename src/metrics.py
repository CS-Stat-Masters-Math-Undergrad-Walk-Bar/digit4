import torch
import torch.nn as nn
import torch.nn.functional as F


def surprise(kl: torch.Tensor, lambda_s: float) -> torch.Tensor:
    kl_clamped = torch.clamp(kl, min=1e-8)
    log_arg = torch.clamp(1 - torch.exp(-lambda_s * kl_clamped), min=1e-8)
    return torch.log(log_arg)


def novelty(
    x_hat: torch.Tensor,
    digit_classifier: nn.Module,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_hat = x_hat.view(-1, 1, 28, 28)
    logits = digit_classifier(x_hat)  # (B, 10)
    probs = F.softmax(logits, dim=1)  # (B, 10)

    p_c1 = probs[:, c1]  # (B,)
    p_c2 = probs[:, c2]  # (B,)

    relevance = p_c1 + p_c2  # (B,)

    p_tilde_c1 = p_c1 / (relevance + eps)  # (B,)
    p_tilde_c2 = p_c2 / (relevance + eps)  # (B,)

    h = -(
        p_tilde_c1 * torch.log(p_tilde_c1 + eps)
        + p_tilde_c2 * torch.log(p_tilde_c2 + eps)
    ) / torch.log(torch.tensor(2.0, device=x_hat.device))  # (B,)

    return (h * relevance).mean()  # scalar


def value(
    x_hat: torch.Tensor,
    digit_vs_nondigit_classifier: nn.Module,
    eps: float = 1e-8,
    emnist_mu: float = 0.1736,
    emnist_sd: float = 0.3317,
) -> torch.Tensor:
    # apply EMNIST normalization
    x_hat = x_hat.view(-1, 1, 28, 28)
    x_norm = (x_hat - emnist_mu) / emnist_sd

    logits = digit_vs_nondigit_classifier(x_norm)  # (B, 1)

    # label convention: 0 = digit, 1 = letter
    p_digit = 1.0 - torch.sigmoid(logits.squeeze(1))  # (B,)

    return torch.log(p_digit + eps).mean()  # scalar


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = value(x_hat, value_classifier, eps)
    n = torch.log(novelty(x_hat, digit_classifier, c1, c2, eps) + eps)
    s = surprise(kl, lambda_s)

    return v, n, s
