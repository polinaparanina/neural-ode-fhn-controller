# Neural ODE Controller for FitzHugh–Nagumo Model

A neural network controller trained end-to-end through a differentiable ODE solver to stabilize the FitzHugh–Nagumo neuron model under a pathological background stimulus.

## Problem

The FitzHugh–Nagumo system under a constant pathological stimulus `z_bg` exhibits undesired oscillatory behavior. The goal is to steer the system to a target equilibrium `(x*, y*)` using a learned control signal `u(x, y, t)` injected into the right-hand side of the ODE.

```
dx/dt = c(y + x − x³/3 + z_bg) + u(x, y, t)
dy/dt = −(1/c)(x − a + by)
```

## Approach

- Controller: 2-layer MLP (input: `[x, y, t]`, output: `u ∈ [−u_max, u_max]`)  
- Solver: RK4 via `torchdiffeq`; gradients flow through the solver  
- Loss: weighted sum of terminal MSE (×1000), control energy (×0.05), and smoothness (×0.5)  
- Training: 500 epochs, Adam, lr=0.005  

MSE converges from ~1e-2 to ~1e-8 over 200 epochs.

## Robustness tests

| Test | Result |
|---|---|
| 15 random initial conditions (radius 1.5) | Convergence in all cases |
| Additive white noise (σ = 0.02) | Stable |
| Ornstein–Uhlenbeck background stimulus | Stable |
| Pulse disturbance (amplitude 0.2, duration 6 s) | Stable |

## Repository structure

```
├── NeuralODE.ipynb   # full pipeline: training, inference, robustness tests
└── requirements.txt
```

## Requirements

```
torch
torchdiffeq
numpy
matplotlib
seaborn
tqdm
```

Install:
```bash
pip install -r requirements.txt
```

## Usage

Open `fhn_controller.ipynb` and run cells sequentially.  
All parameters are defined in the `PARAMS` dict at the top of the notebook.
