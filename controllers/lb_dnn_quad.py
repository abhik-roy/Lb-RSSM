import os
import numpy as np
import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):
    def __init__(self, phi_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, phi_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class LbDNNQuad:
    Kp_pos = np.diag([3.0, 3.0, 5.0])
    Kd_pos = np.diag([2.5, 2.5, 3.5])
    Kp_att = np.diag([30.0, 30.0, 15.0])
    Kd_att = np.diag([ 6.0,  6.0,  4.0])

    def __init__(self, phi_dim=48, gamma=0.05, pretrained_path=None):
        self.phi_dim = phi_dim
        self.gamma   = gamma
        self.m       = 0.5
        self.g       = 9.81

        self.encoder = FrozenEncoder(phi_dim=phi_dim)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.W_hat = np.zeros((3, phi_dim))

        if pretrained_path and os.path.exists(pretrained_path):
            self.encoder.load_state_dict(
                torch.load(pretrained_path, weights_only=True))
            print(f"Loaded encoder: {pretrained_path}")
            w_path = pretrained_path.replace(".pt", "_w_hat.npy")
            if os.path.exists(w_path):
                self.W_hat = np.load(w_path)
                print(f"Loaded W_hat:   {w_path}")

    def phi(self, pos, vel):
        x = torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)
        with torch.no_grad():
            return self.encoder(x).numpy()

    def outer_loop(self, state, pd, vd, ad, phi_vec):
        pos = state[:3]; vel = state[3:6]
        e_pos = pos - pd; e_vel = vel - vd

        f_nn  = self.W_hat @ phi_vec
        a_des = ad - self.Kd_pos @ e_vel - self.Kp_pos @ e_pos + f_nn / self.m

        T = float(np.clip(self.m * (self.g + a_des[2]), 0.1, 20.0))
        # R@[0,0,T] ≈ [T·sin(θ), -T·sin(φ), T]:  +θ → +ax,  -φ → +ay
        phi_d   = float(np.clip(-a_des[1] / self.g, -0.3, 0.3))
        theta_d = float(np.clip( a_des[0] / self.g, -0.3, 0.3))
        return T, np.array([phi_d, theta_d, 0.0]), e_pos, e_vel

    def inner_loop(self, state, eta_d):
        return -(self.Kp_att @ (state[6:9] - eta_d) + self.Kd_att @ state[9:12])

    def compute_control(self, state, t, quad):
        phi_vec = self.phi(state[:3], state[3:6])
        pd, vd, ad = quad.figure8(t)
        T, eta_d, e_pos, e_vel = self.outer_loop(state, pd, vd, ad, phi_vec)
        tau = self.inner_loop(state, eta_d)
        return T, tau, e_pos, e_vel, phi_vec

    def update_weights(self, phi_vec, e_vel, dt):
        self.W_hat += self.gamma * np.outer(e_vel, phi_vec) * dt
