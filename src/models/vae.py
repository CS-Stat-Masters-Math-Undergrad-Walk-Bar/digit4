import torch
import torch.nn as nn

from typing import Sequence, Literal


def he_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        real_dim: int,
        h_dims: int | Sequence[int],
        bn_dim: int,
        activation: nn.Module = nn.ReLU(),
        decoder_type: Literal["gaussian", "bernoulli"] = "gaussian",
        initialize: bool = False,
    ):
        super().__init__()

        if isinstance(h_dims, int):
            h_dims = [h_dims]
        else:
            h_dims = list(h_dims)

        # ------- ENCODER -------
        encoder_shared_layers = []
        prev_dim = real_dim
        for h in h_dims:
            encoder_shared_layers.append(nn.Linear(prev_dim, h))
            encoder_shared_layers.append(type(activation)())
            prev_dim = h

        self.Encoder = nn.Sequential(*encoder_shared_layers)
        self.mu_head = nn.Linear(prev_dim, bn_dim)
        self.log_var_head = nn.Linear(prev_dim, bn_dim)  # outputs log sigma sq

        # ------- DECODER -------
        decoder_layers = []
        prev_dim = bn_dim
        for h in h_dims[::-1]:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(type(activation)())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, real_dim))
        if decoder_type == "bernoulli":
            decoder_layers.append(nn.Sigmoid())

        self.Decoder = nn.Sequential(*decoder_layers)

        # ------- INITIALIZATIONS -------
        if initialize:
            self.apply(he_initialization)

            if decoder_type == "bernoulli":
                nn.init.xavier_uniform_(self.Decoder[-2].weight)
                nn.init.zeros_(self.Decoder[-2].bias)

    def encode(self, x: torch.Tensor):
        shared = self.Encoder(x)
        mu = self.mu_head(shared)
        log_var = self.log_var_head(shared)

        return mu, log_var

    def decode(self, X: torch.Tensor):
        return self.Decoder(X)

    def reparamaterize(self, mu: torch.Tensor, log_var: torch.tensor):
        var = torch.exp(0.5 * log_var)
        eps = torch.randn_like(var)
        out = mu + var * eps

        return out

    def forward(self, X):
        mu, log_var = self.encode(X)
        z = self.reparamaterize(mu, log_var)
        x_hat = self.decode(z)

        return x_hat, mu, log_var
