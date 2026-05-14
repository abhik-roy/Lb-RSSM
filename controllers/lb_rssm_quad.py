import os
import numpy as np
import torch

from controllers.rssm_core import RSSMCore


class LbRSSMQuad:
    Kp_pos = np.diag([3.0, 3.0, 5.0])
    Kd_pos = np.diag([2.5, 2.5, 3.5])
    Kp_att = np.diag([30.0, 30.0, 15.0])
    Kd_att = np.diag([ 6.0,  6.0,  4.0])

    def __init__(self, gru_dim=32, latent_dim=16, gamma=0.05,
                 pretrained_path=None):
        self.phi_dim    = gru_dim + latent_dim
        self.gru_dim    = gru_dim
        self.latent_dim = latent_dim
        self.gamma      = gamma
        self.m          = 0.5
        self.g          = 9.81

        self.rssm = RSSMCore(input_dim=6, gru_dim=gru_dim, latent_dim=latent_dim)
        for p in self.rssm.parameters():
            p.requires_grad_(False)

        self.h_t   = torch.zeros(1, gru_dim)
        self.W_hat = np.zeros((3, self.phi_dim))

        self.sigma_scalar = 1.0
        self.ball         = 1.0

        if pretrained_path and os.path.exists(pretrained_path):
            self.rssm.load_state_dict(
                torch.load(pretrained_path, weights_only=True))
            print(f"Loaded RSSM:  {pretrained_path}")
            w_path = pretrained_path.replace(".pt", "_w_hat.npy")
            if os.path.exists(w_path):
                self.W_hat = np.load(w_path)
                print(f"Loaded W_hat: {w_path}")

    def phi(self, pos, vel):
        x = torch.tensor(
            np.concatenate([pos, vel]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h_new, mu, sigma = self.rssm(x, self.h_t)
        self.h_t = h_new

        z_t     = (mu + sigma * torch.randn_like(mu)).squeeze(0)
        phi_vec = torch.cat([h_new.squeeze(0), z_t]).numpy()

        self.sigma_scalar = sigma.squeeze(0).mean().item()
        self.ball = self.sigma_scalar / np.sqrt(max(2.5 - 0.5, 1e-3))
        return phi_vec

    def outer_loop(self, state, pd, vd, ad, phi_vec):
        pos = state[:3]; vel = state[3:6]
        e_pos = pos - pd; e_vel = vel - vd

        f_nn  = self.W_hat @ phi_vec
        a_des = ad - self.Kd_pos @ e_vel - self.Kp_pos @ e_pos + f_nn / self.m

        T       = float(np.clip(self.m * (self.g + a_des[2]), 0.1, 20.0))
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

    def reset(self):
        self.W_hat        = np.zeros((3, self.phi_dim))
        self.h_t          = torch.zeros(1, self.gru_dim)
        self.sigma_scalar = 1.0
        self.ball         = 1.0
