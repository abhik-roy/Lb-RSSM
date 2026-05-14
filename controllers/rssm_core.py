import torch
import torch.nn as nn


class RSSMCore(nn.Module):
    def __init__(self, input_dim=6, gru_dim=32, latent_dim=16):
        super().__init__()
        self.gru_dim    = gru_dim
        self.latent_dim = latent_dim
        self.gru     = nn.GRUCell(input_dim, gru_dim)
        self.encoder = nn.Sequential(
            nn.Linear(gru_dim, gru_dim), nn.Tanh(),
            nn.Linear(gru_dim, latent_dim * 2),
        )

    def forward(self, x, h_prev):
        x_in = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
        h_in = h_prev.view(-1, h_prev.shape[-1]) if h_prev.dim() > 2 else h_prev
        h_t  = self.gru(x_in, h_in)
        mu, log_sigma = self.encoder(h_t).chunk(2, dim=-1)
        sigma = torch.exp(log_sigma.clamp(-4, 2))
        return h_t, mu, sigma
