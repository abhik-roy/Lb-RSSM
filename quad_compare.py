import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from quad_dynamics import Quadrotor
from controllers.lb_dnn_quad  import LbDNNQuad
from controllers.lb_rssm_quad import LbRSSMQuad

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def run(ctrl, quad, T_sim=30.0, dt=0.001):
    p0, v0, _ = quad.figure8(0.0)
    state = np.array([p0[0], p0[1], p0[2],
                      v0[0], v0[1], v0[2],
                      0., 0., 0., 0., 0., 0.])
    steps = int(T_sim / dt)
    log   = dict(t=[], e_pos=[], wind_true=[], wind_est=[],
                 sigma=[], ball=[])

    for i in range(steps):
        t = i * dt
        T_thrust, tau, e_pos, e_vel, phi_vec = ctrl.compute_control(state, t, quad)
        ctrl.update_weights(phi_vec, e_vel, dt)
        state = quad.step(state, T_thrust, tau, t)
        if not np.all(np.isfinite(state)):
            state = np.clip(state, -50, 50); state[np.isnan(state)] = 0.

        log["t"].append(t)
        log["e_pos"].append(e_pos.copy())
        log["wind_true"].append(quad.wind(t, state[:3]).copy())
        log["wind_est"].append(-(ctrl.W_hat @ phi_vec).copy())
        log["sigma"].append(getattr(ctrl, "sigma_scalar", 0.0))
        log["ball"].append(getattr(ctrl, "ball", 0.0))

    return {k: np.array(v) for k, v in log.items()}


def plot_comparison(logs):
    dnn  = logs["Lb-DNN"]
    rssm = logs["Lb-RSSM"]
    t    = dnn["t"]

    err_dnn  = np.linalg.norm(dnn["e_pos"],  axis=1)
    err_rssm = np.linalg.norm(rssm["e_pos"], axis=1)

    rms_dnn  = np.sqrt(np.mean(err_dnn[-5000:]**2))
    rms_rssm = np.sqrt(np.mean(err_rssm[-5000:]**2))

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    CDNN  = "#2196F3"
    CRSSM = "#FF5722"

    # ── Row 0: error — DNN full scale (left), RSSM zoomed (right) ────────────
    ax_dnn = fig.add_subplot(gs[0, 0])
    ax_dnn.plot(t, err_dnn, color=CDNN, lw=1.4)
    ax_dnn.set_ylabel("||e_pos|| (m)")
    ax_dnn.set_title(f"Lb-DNN  (RMS = {rms_dnn:.1f} m — diverges)",
                     color=CDNN, fontweight="bold")
    ax_dnn.set_xlim(0, t[-1])

    ax_rssm = fig.add_subplot(gs[0, 1])
    ax_rssm.plot(t, err_rssm, color=CRSSM, lw=1.4)
    ax_rssm.axhline(0.15, color="k", lw=0.8, linestyle=":", alpha=0.5,
                    label="0.15 m")
    ax_rssm.set_ylabel("||e_pos|| (m)")
    ax_rssm.set_title(f"Lb-RSSM  (RMS = {rms_rssm:.3f} m — bounded)",
                      color=CRSSM, fontweight="bold")
    ax_rssm.set_xlim(0, t[-1])
    ax_rssm.legend(fontsize=8)

    # ── Row 1: wind Fx estimation ─────────────────────────────────────────────
    for col, (name, log, color) in enumerate([
            ("Lb-DNN",  dnn,  CDNN),
            ("Lb-RSSM", rssm, CRSSM)]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t, log["wind_true"][:, 0], "k--", lw=1.1, alpha=0.6,
                label="True Fx")
        ax.plot(t, log["wind_est"][:,  0], color=color, lw=1.3,
                label=f"{name} estimate")
        # clip y so DNN's runaway estimate doesn't crush the signal
        y_max = max(log["wind_true"][:, 0].max() * 2.5, 1.5)
        ax.set_ylim(-y_max, y_max)
        ax.set_ylabel("Force (N)")
        ax.set_title(f"{name} — wind Fx estimation")
        ax.legend(fontsize=8)
        ax.set_xlim(0, t[-1])

    # ── Row 2: overlay on capped axis (left) + RSSM σ_t (right) ─────────────
    cap = min(err_rssm.max() * 6, 3.0)
    ax0 = fig.add_subplot(gs[2, 0])
    ax0.plot(t, np.clip(err_dnn, 0, cap * 1.05),
             color=CDNN, lw=1.3, label=f"DNN (→ {rms_dnn:.0f} m RMS)")
    ax0.plot(t, err_rssm, color=CRSSM, lw=1.3, label=f"RSSM ({rms_rssm:.2f} m RMS)")
    ax0.axhline(cap, color=CDNN, lw=0.7, linestyle=":", alpha=0.6,
                label="DNN off-scale ↑")
    ax0.set_ylim(0, cap * 1.1)
    ax0.set_ylabel("Error (m)"); ax0.set_xlabel("Time (s)")
    ax0.set_title("Overlay — DNN clipped at plot ceiling")
    ax0.legend(fontsize=8); ax0.set_xlim(0, t[-1])

    ax1 = fig.add_subplot(gs[2, 1])
    ax1.plot(t, rssm["sigma"], color="#FF9800", lw=1.3, label="σ_t (uncertainty)")
    ax1.plot(t, rssm["ball"],  color="#9C27B0", lw=1.1, linestyle="--",
             label="Lyapunov ball")
    ax1.plot(t, err_rssm, color=CRSSM, lw=1.0, alpha=0.6, label="||e||")
    ax1.set_ylabel("Value"); ax1.set_xlabel("Time (s)")
    ax1.set_title("RSSM: σ_t uncertainty vs actual error")
    ax1.legend(fontsize=8); ax1.set_xlim(0, t[-1])

    plt.suptitle("Lb-DNN vs Lb-RSSM — purely temporal wind\n"
                 "φ(pos, vel) carries no temporal information → DNN W* never converges",
                 fontsize=12, fontweight="bold")
    path = os.path.join(OUT, "quad_compare_temporal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    quad = Quadrotor(dt=0.001, wind_scale=1.0)

    controllers = {
        "Lb-DNN": LbDNNQuad(
            phi_dim=48, gamma=0.05,
            pretrained_path="quad_encoder_pretrained.pt"),
        "Lb-RSSM": LbRSSMQuad(
            gru_dim=32, latent_dim=16, gamma=0.05,
            pretrained_path="quad_rssm_pretrained.pt"),
    }

    logs = {}
    for name, ctrl in controllers.items():
        print(f"Running {name}...")
        logs[name] = run(ctrl, quad, T_sim=30.0, dt=0.001)
        err = np.linalg.norm(logs[name]["e_pos"], axis=1)
        print(f"  RMS (tail 5s): {np.sqrt(np.mean(err[-5000:]**2)):.4f} m  "
              f"max: {err.max():.4f} m")

    plot_comparison(logs)
    print("Done.")
