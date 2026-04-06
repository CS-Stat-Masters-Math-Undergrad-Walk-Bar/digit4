import torch
import torch.nn.functional as F
from typing import Sequence, Literal


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
