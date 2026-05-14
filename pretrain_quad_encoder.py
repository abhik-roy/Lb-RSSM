import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from quad_dynamics import Quadrotor
from controllers.lb_dnn_quad import FrozenEncoder


def collect_data(n_episodes=200, T=12.0, dt=0.005):
    quad  = Quadrotor(dt=dt, wind_scale=1.0)
    rng   = np.random.default_rng(0)
    data  = []
    steps = int(T / dt)

    for ep in range(n_episodes):
        t_start = rng.uniform(0, 15.7)
        p0, v0, _ = quad.figure8(t_start)
        state = np.array([
            p0[0] + rng.uniform(-0.9, 0.9),
            p0[1] + rng.uniform(-0.7, 0.7),
            p0[2] + rng.uniform(-0.3, 0.3),
            v0[0] + rng.uniform(-0.5, 0.5),
            v0[1] + rng.uniform(-0.5, 0.5),
            rng.uniform(-0.2, 0.2),
            0., 0., 0., 0., 0., 0.
        ])
        Kp = rng.uniform(2.0, 8.0)
        Kd = rng.uniform(1.5, 5.0)

        for i in range(steps):
            t   = t_start + i * dt
            pos = state[:3]; vel = state[3:6]; eta = state[6:9]
            pd, vd, _ = quad.figure8(t)
            a_des    = -Kd*(vel - vd) - Kp*(pos - pd)
            T_thrust = float(np.clip(quad.m*(quad.g + a_des[2]), 0.5, 18.0))
            phi_d    = float(np.clip(-a_des[1]/quad.g, -0.45, 0.45))
            theta_d  = float(np.clip( a_des[0]/quad.g, -0.45, 0.45))
            tau = np.array([
                -(25.0*(eta[0] - phi_d)   + 5.0*state[9]),
                -(25.0*(eta[1] - theta_d) + 5.0*state[10]),
                -(15.0* eta[2]            + 3.0*state[11]),
            ])
            data.append((
                np.concatenate([pos, vel]).astype(np.float32),
                quad.wind(t, pos).astype(np.float32),
            ))
            state = quad.step(state, T_thrust, tau, t)
            if not np.all(np.isfinite(state)):
                state = np.clip(state, -20, 20); state[np.isnan(state)] = 0.

    return data


class EncoderWithHead(nn.Module):
    def __init__(self, phi_dim=48):
        super().__init__()
        self.encoder = FrozenEncoder(phi_dim=phi_dim)
        self.head    = nn.Linear(phi_dim, 3, bias=False)

    def forward(self, x):
        phi = self.encoder(x)
        return phi, self.head(phi)


def train(n_epochs=30, phi_dim=48, save_path="quad_encoder_pretrained.pt"):
    print("Collecting flight data...")
    data = collect_data(n_episodes=200, T=12.0, dt=0.005)
    print(f"  {len(data):,} transitions")

    xs = torch.tensor([d[0] for d in data])
    ys = torch.tensor([d[1] for d in data])

    model     = EncoderWithHead(phi_dim=phi_dim)
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    n, batch  = len(data), 512

    print("Training encoder...")
    for epoch in range(n_epochs):
        idx = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            b       = idx[i:i+batch]
            _, pred = model(xs[b])
            loss    = ((pred - ys[b])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={total/n:.5f}")

    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder → {save_path}")

    w_path = save_path.replace(".pt", "_w_hat.npy")
    # W_hat compensates wind (f_nn = -wind), head was trained to predict +wind
    np.save(w_path, -model.head.weight.detach().numpy())
    print(f"Saved W_hat   → {w_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    train(save_path=os.path.join(base, "quad_encoder_pretrained.pt"))
