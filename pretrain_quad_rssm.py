import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quad_dynamics import Quadrotor
from controllers.rssm_core import RSSMCore


def collect_episodes(n_episodes=200, T=15.0, dt=0.005):
    quad  = Quadrotor(dt=dt, wind_scale=1.0)
    rng   = np.random.default_rng(0)
    steps = int(T / dt)
    episodes = []

    for ep in range(n_episodes):
        t_start = rng.uniform(0, 15.7)
        p0, v0, _ = quad.figure8(t_start)
        state = np.array([
            p0[0] + rng.uniform(-0.7, 0.7),
            p0[1] + rng.uniform(-0.5, 0.5),
            p0[2] + rng.uniform(-0.25, 0.25),
            v0[0] + rng.uniform(-0.4, 0.4),
            v0[1] + rng.uniform(-0.4, 0.4),
            rng.uniform(-0.15, 0.15),
            0., 0., 0., 0., 0., 0.
        ])
        Kp = rng.uniform(2.5, 7.0)
        Kd = rng.uniform(1.5, 4.5)
        ep_xs, ep_ys = [], []

        for i in range(steps):
            t   = t_start + i * dt
            pos = state[:3]; vel = state[3:6]; eta = state[6:9]
            pd, vd, _ = quad.figure8(t)
            a_des    = -Kd*(vel-vd) - Kp*(pos-pd)
            T_thrust = float(np.clip(quad.m*(quad.g+a_des[2]), 0.5, 18.0))
            phi_d    = float(np.clip(-a_des[1]/quad.g, -0.45, 0.45))
            theta_d  = float(np.clip( a_des[0]/quad.g, -0.45, 0.45))
            tau = np.array([
                -(25.*(eta[0]-phi_d)   + 5.*state[9]),
                -(25.*(eta[1]-theta_d) + 5.*state[10]),
                -(15.* eta[2]          + 3.*state[11]),
            ])
            ep_xs.append(np.concatenate([pos, vel]).astype(np.float32))
            ep_ys.append(quad.wind(t, pos).astype(np.float32))
            state = quad.step(state, T_thrust, tau, t)
            if not np.all(np.isfinite(state)):
                state = np.clip(state, -20, 20); state[np.isnan(state)] = 0.

        episodes.append((np.array(ep_xs), np.array(ep_ys)))

    return episodes


class RSSMSeqTrainer(nn.Module):
    """nn.GRU processes full episode sequences in one vectorised call."""

    def __init__(self, gru_dim=32, latent_dim=16):
        super().__init__()
        self.gru_dim    = gru_dim
        self.latent_dim = latent_dim
        self.gru     = nn.GRU(6, gru_dim, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(gru_dim, gru_dim), nn.Tanh(),
            nn.Linear(gru_dim, latent_dim * 2),
        )
        self.head = nn.Linear(gru_dim + latent_dim, 3, bias=False)

    def forward(self, x_batch):
        h_all, _ = self.gru(x_batch)            # (B, T, gru_dim)
        B, T, D  = h_all.shape
        h_flat   = h_all.reshape(B * T, D)
        mu, log_sigma = self.encoder(h_flat).chunk(2, dim=-1)
        sigma    = torch.exp(log_sigma.clamp(-4, 2))
        z        = mu + sigma * torch.randn_like(mu)
        phi      = torch.cat([h_flat, z], dim=-1)
        return self.head(phi), mu, sigma

    def export_rssm_core(self):
        core = RSSMCore(input_dim=6, gru_dim=self.gru_dim,
                        latent_dim=self.latent_dim)
        core.gru.weight_ih.data = self.gru.weight_ih_l0.data.clone()
        core.gru.weight_hh.data = self.gru.weight_hh_l0.data.clone()
        core.gru.bias_ih.data   = self.gru.bias_ih_l0.data.clone()
        core.gru.bias_hh.data   = self.gru.bias_hh_l0.data.clone()
        core.encoder.load_state_dict(self.encoder.state_dict())
        return core


def train(n_epochs=40, gru_dim=32, latent_dim=16,
          batch_eps=8, save_path="quad_rssm_pretrained.pt"):

    print("Collecting episodes...")
    episodes = collect_episodes(n_episodes=200, T=15.0, dt=0.005)
    print(f"  {len(episodes)} episodes × {episodes[0][0].shape[0]} steps")

    xs = torch.tensor(np.stack([e[0] for e in episodes]))
    ys = torch.tensor(np.stack([e[1] for e in episodes]))
    N  = xs.shape[0]

    model     = RSSMSeqTrainer(gru_dim=gru_dim, latent_dim=latent_dim)
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    print("Training RSSM (sequential episodes)...")
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        total_loss = 0.0; total_pts = 0

        for i in range(0, N, batch_eps):
            b   = perm[i : i + batch_eps]
            x_b = xs[b]; y_b = ys[b]
            B, T, _ = x_b.shape

            pred, mu, sigma = model(x_b)
            y_flat = y_b.reshape(B * T, 3)

            recon = ((pred - y_flat)**2).mean()
            kl    = (0.5*(mu**2 + sigma**2 - 2*sigma.log() - 1)).mean()
            loss  = recon + 0.05 * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * (B * T)
            total_pts  += B * T

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={total_loss/total_pts:.5f}")

    core = model.export_rssm_core()
    torch.save(core.state_dict(), save_path)
    print(f"Saved RSSM   → {save_path}")

    w_path = save_path.replace(".pt", "_w_hat.npy")
    np.save(w_path, -model.head.weight.detach().numpy())
    print(f"Saved W_hat  → {w_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    train(save_path=os.path.join(base, "quad_rssm_pretrained.pt"))
