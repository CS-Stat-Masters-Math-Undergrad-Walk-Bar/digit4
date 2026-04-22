import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Literal
from .metrics import log_creativity_score


def ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    # decoder_dist: Literal["gaussian", "bernoulli"],
):
    x_hat, mu, log_var = pred

    kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()

    # if decoder_dist == "bernoulli":
    #     reconstruction = -F.binary_cross_entropy(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()
    # elif decoder_dist == "gaussian":
    #     reconstruction = -F.mse_loss(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()

    return reconstruction, kl


# def neg_ELBO(
#     pred: Sequence[torch.Tensor],
#     x: torch.Tensor,
# ):
#     return -ELBO(pred, x)


def creative_ELBO(
    pred: Sequence[torch.Tensor],
    x: torch.Tensor,
    # decoder_dist: Literal["gaussian", "bernoulli"],
    digit_classifier: nn.Module,
    value_classifier: nn.Module,
    value_weight: float,
    novelty_weight: float,
    surprise_weight: float,
    lambda_s: float,
    c1: int = 2,
    c2: int = 6,
    eps: float = 1e-8,

) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    x_hat, mu, log_var = pred

    # if decoder_dist == "bernoulli":
    #     reconstruction = -F.binary_cross_entropy(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()
    # elif decoder_dist == "gaussian":
    #     reconstruction = -F.mse_loss(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()

    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="none").view(x_hat.shape[0], -1).sum(dim=1).mean()


    kl = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()

    v, n, s = log_creativity_score(
        x_hat,
        kl,
        digit_classifier,
        value_classifier,
        value_weight,
        novelty_weight,
        surprise_weight,
        lambda_s,
        c1,
        c2,
        eps,
    )



    return reconstruction, kl, v, n, s


# def neg_creative_ELBO(
#     pred: Sequence[torch.Tensor],
#     x: torch.Tensor,
#     decoder_dist: Literal["gaussian", "bernoulli"],
#     digit_classifier: nn.Module,
#     value_classifier: nn.Module,
#     value_weight: float,
#     novelty_weight: float,
#     surprise_weight: float,
#     lambda_s: float,
#     c1: int = 2,
#     c2: int = 6,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     return -creative_ELBO(
#         pred,
#         x,
#         decoder_dist,
#         digit_classifier,
#         value_classifier,
#         value_weight,
#         novelty_weight,
#         surprise_weight,
#         lambda_s,
#         c1,
#         c2,
#         eps,
#     )
