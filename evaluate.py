"""
Validate the trained PINN against the numerical (solve_bvp) ground truth ---
there is no closed-form solution to fall back on for this nonlinear problem,
so this IS the ground truth.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as cfg
from .numerical_baseline import solve_single
from .pinn_model import BeamPINN


def load_model():
    model = BeamPINN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(cfg.RESULTS_DIR, "pinn_model.pt"), map_location=cfg.DEVICE))
    model.eval()
    return model


def plot_loss_curve(history):
    plt.figure(figsize=(6, 4))
    plt.semilogy(history["epoch"], history["total"], label="total loss")
    plt.semilogy(history["epoch"], history["pde"], label="PDE residual loss", alpha=0.7)
    plt.semilogy(history["epoch"], history["bc"], label="boundary loss", alpha=0.7)
    plt.xlabel("epoch")
    plt.ylabel("loss (log scale)")
    plt.title("PINN training convergence (nonlinear foundation)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(cfg.RESULTS_DIR, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def validate_against_numerical(model, test_cases=None):
    if test_cases is None:
        test_cases = [
            (cfg.E_NOM, cfg.Q_NOM, "nominal (E0, q0)"),
            (0.8 * cfg.E_NOM, cfg.Q_NOM, "soft material (0.8*E0)"),
            (1.2 * cfg.E_NOM, cfg.Q_NOM, "stiff material (1.2*E0)"),
            (cfg.E_NOM, 1.5 * cfg.Q_NOM, "high load (1.5*q0)"),
            (cfg.E_NOM, 2.0 * cfg.Q_NOM, "extreme load (2.0*q0)"),
        ]

    plt.figure(figsize=(7, 5))
    errors = {}
    for E, q, label in test_cases:
        x_np, w_true = solve_single(E, q, return_full=True)
        x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1)
        E_t = torch.full_like(x_t, E)
        q_t = torch.full_like(x_t, q)
        with torch.no_grad():
            w_pinn = model(x_t, E_t, q_t).numpy().flatten()

        rel_l2 = np.linalg.norm(w_pinn - w_true) / (np.linalg.norm(w_true) + 1e-12)
        errors[label] = float(rel_l2)

        plt.plot(x_np, w_true * 1000, "-", label=f"numerical BVP: {label}")
        plt.plot(x_np, w_pinn * 1000, "--", label=f"PINN: {label}")

    plt.xlabel("x [m]")
    plt.ylabel("deflection w(x) [mm]")
    plt.title("PINN vs. numerical ground truth (nonlinear foundation)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(cfg.RESULTS_DIR, "pinn_vs_numerical.png")
    plt.savefig(path, dpi=150)
    plt.close()

    with open(os.path.join(cfg.RESULTS_DIR, "validation_errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    return errors, path


if __name__ == "__main__":
    model = load_model()
    with open(os.path.join(cfg.RESULTS_DIR, "training_history.json")) as f:
        history = json.load(f)
    plot_loss_curve(history)
    errors, _ = validate_against_numerical(model)
    print(json.dumps(errors, indent=2))
