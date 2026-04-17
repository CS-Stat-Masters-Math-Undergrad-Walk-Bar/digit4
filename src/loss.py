import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Literal
from .metrics import creativity_score


def ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    decoder_dist: Literal["gaussian", "bernoulli"],
):
    x_hat, mu, log_var = pred

    kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()

    if decoder_dist == "bernoulli":
        reconstruction = -F.binary_cross_entropy(x_hat, x, reduction="mean")
    elif decoder_dist == "gaussian":
        reconstruction = -F.mse_loss(x_hat, x, reduction="mean")

    return reconstruction - kl


def neg_ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    decoder_dist: Literal["gaussian", "bernoulli"],
):
    return -ELBO(pred, x, decoder_dist)


def creative_ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    decoder_dist: Literal["gaussian", "bernoulli"],
    digit_classifier: nn.Module,
    value_classifier: nn.Module,
    a1: float,
    a2: float,
    a3: float,
    lambda_s: float,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:

    x_hat, mu, log_var = pred

    if decoder_dist == "bernoulli":
        reconstruction = -F.binary_cross_entropy(x_hat, x, reduction="mean")
    elif decoder_dist == "gaussian":
        reconstruction = -F.mse_loss(x_hat, x, reduction="mean")

    kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()

    log_dc = creativity_score(
        x_hat,
        kl,
        digit_classifier,
        value_classifier,
        a1,
        a2,
        a3,
        lambda_s,
        c1,
        c2,
        eps,
    )

    return reconstruction + log_dc


def neg_creative_ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    decoder_dist: Literal["gaussian", "bernoulli"],
    digit_classifier: nn.Module,
    value_classifier: nn.Module,
    a1: float,
    a2: float,
    a3: float,
    lambda_s: float,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    return -creative_ELBO(
        pred,
        x,
        decoder_dist,
        digit_classifier,
        value_classifier,
        a1,
        a2,
        a3,
        lambda_s,
        c1,
        c2,
        eps,
    )
