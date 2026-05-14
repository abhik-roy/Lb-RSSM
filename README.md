# Lb-RSSM

An extension of [Patil et al.'s Lyapunov-based DNN (Lb-DNN)](https://faculty.sites.ufl.edu/npatil/research/) controller that replaces the memoryless feature encoder with a Recurrent State-Space Model (RSSM). The key result: Lb-DNN diverges under time-varying disturbances that Lb-RSSM handles stably.

## The problem with Lb-DNN under temporal disturbances

The Lb-DNN stability proof (Patil et al.) guarantees UUB error provided the optimal weight matrix **W\*** stays approximately constant. This holds when the disturbance depends only on position — the same state always maps to the same wind force, so a memoryless encoder can learn it.

With a **purely temporal disturbance**, `φ(pos, vel)` carries zero information about the current wind. The online update rule pushes **Ŵ** toward the instantaneous wind at every step, but the target changes every step with no correlation to the features — **Ŵ** converges to the time-average (≈ 0) and the residual error equals the full wind magnitude. Closed-loop, this means the PD + DNN system is effectively PD-only against an uncompensated ~1 N disturbance, which destabilizes it.

The RSSM fixes this by including the GRU hidden state `h_t` in the feature vector `φ_t = [h_t, z_t]`. `h_t` is updated every step from the sequence of `[pos, vel]` observations, implicitly tracking where the drone is in the temporal wind cycle. The mapping from `φ_t` to wind force is now approximately stationary, restoring the Lyapunov convergence condition.

## Results

Wind: `Fx = 0.8·sin(0.4t)`,  `Fy = 0.6·cos(0.3t)`,  `Fz = 0.35·sin(0.5t)`  — purely temporal, no spatial component.

| Controller | RMS error (tail 5 s) | Max error |
|------------|----------------------|-----------|
| Lb-DNN     | 111.5 m *(diverges)* | 136.5 m   |
| Lb-RSSM    | **0.43 m**           | **0.87 m** |

![Comparison](figures/quad_compare_temporal.png)

## Getting started

```bash
pip install -r requirements.txt
```

Pretrained weights are included. To reproduce training from scratch:

```bash
python3 pretrain_quad_encoder.py   # ~2 min
python3 pretrain_quad_rssm.py      # ~3 min
```

Run the comparison:

```bash
python3 quad_compare.py
# → figures/quad_compare_temporal.png
```

## File structure

```
quad_dynamics.py            quadrotor physics + wind model
quad_compare.py             side-by-side DNN vs RSSM simulation

controllers/
  rssm_core.py              GRUCell + stochastic encoder (RSSMCore)
  lb_dnn_quad.py            memoryless Lb-DNN controller
  lb_rssm_quad.py           recurrent Lb-RSSM controller

pretrain_quad_encoder.py    offline pretraining for the DNN encoder
pretrain_quad_rssm.py       sequential episode pretraining for the RSSM
```

## Architecture

```
Lb-DNN:   φ(x) = MLP([pos, vel])               → stateless, can't track wind phase
Lb-RSSM:  h_t  = GRU(h_{t-1}, [pos, vel])
          z_t  ~ N(μ_t, σ_t)  from MLP(h_t)
          φ_t  = [h_t, z_t]                     → h_t encodes temporal context
```

`σ_t` gives a live uncertainty estimate. The theoretical Lyapunov error ball scales as `σ_t / √(K_d − 0.5)` — it widens when the drone enters an unfamiliar part of the wind field.
