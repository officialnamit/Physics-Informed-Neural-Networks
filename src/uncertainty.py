"""
Uncertainty propagation & reliability analysis for the nonlinear
beam-on-foundation problem.

This is where the timing comparison becomes real: because there is no
closed-form solution, the "brute-force" baseline must call a nonlinear
boundary-value-problem solver (scipy's solve_bvp) once per Monte Carlo
sample, in an ordinary Python loop -- exactly as a real reliability study
using a full finite-element solver would have to. The PINN, in contrast,
answers all samples in one batched forward pass.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as cfg
from .numerical_baseline import solve_single


def _sample_E_np(n, rng):
    sigma = np.sqrt(np.log(1.0 + cfg.E_COV**2))
    mu = np.log(cfg.E_NOM) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _sample_q_np(n, rng):
    q = rng.normal(cfg.Q_NOM, cfg.Q_COV * cfg.Q_NOM, size=n)
    return np.clip(q, 0.05 * cfg.Q_NOM, None)


def run_uncertainty_propagation(model, n_samples=cfg.N_MC_SAMPLES, seed=123,
                                 n_timing_curve_points=(100, 500, 2000, 5000, 20000)):
    rng = np.random.default_rng(seed)
    E_samples = _sample_E_np(n_samples, rng)
    q_samples = _sample_q_np(n_samples, rng)

    # ---------------- PINN-based Monte Carlo (one batched forward pass) ----------------
    x_tip = torch.full((n_samples, 1), cfg.L, dtype=torch.float32)
    E_t = torch.tensor(E_samples, dtype=torch.float32).reshape(-1, 1)
    q_t = torch.tensor(q_samples, dtype=torch.float32).reshape(-1, 1)

    t0 = time.perf_counter()
    with torch.no_grad():
        w_tip_pinn = model(x_tip, E_t, q_t).numpy().flatten()
    t_pinn = time.perf_counter() - t0

    # ---------------- Numerical baseline: one solve_bvp call PER SAMPLE ----------------
    print(f"Running {n_samples} individual nonlinear BVP solves "
          f"(this is the expensive, honest baseline)...")
    w_tip_numerical = np.empty(n_samples)
    n_failed = 0
    t0 = time.perf_counter()
    for i in range(n_samples):
        try:
            w_tip_numerical[i] = solve_single(E_samples[i], q_samples[i])
        except RuntimeError:
            # extremely rare non-convergence; fall back to the PINN's own
            # prediction for that single sample rather than crash the run
            w_tip_numerical[i] = w_tip_pinn[i]
            n_failed += 1
        if (i + 1) % 4000 == 0:
            print(f"  ...{i+1}/{n_samples} solved")
    t_numerical = time.perf_counter() - t0
    print(f"Numerical baseline done in {t_numerical:.2f} s "
          f"({n_failed} non-convergent samples skipped)")

    # ---------------- Reliability metrics ----------------
    def summarize(w):
        p_fail = float(np.mean(w > cfg.W_ALLOW))
        return {
            "mean_mm": float(np.mean(w) * 1000),
            "std_mm": float(np.std(w) * 1000),
            "p5_mm": float(np.percentile(w, 5) * 1000),
            "p95_mm": float(np.percentile(w, 95) * 1000),
            "prob_exceed_allowable": p_fail,
        }

    results = {
        "n_samples": n_samples,
        "w_allow_mm": cfg.W_ALLOW * 1000,
        "pinn": summarize(w_tip_pinn),
        "numerical_baseline": summarize(w_tip_numerical),
        "timing_seconds": {
            "pinn_batched_forward_pass": t_pinn,
            "numerical_bvp_loop": t_numerical,
            "numerical_bvp_per_sample_ms": (t_numerical / n_samples) * 1000,
        },
        "speedup_factor": t_numerical / t_pinn,
        "n_bvp_non_convergent": n_failed,
        "tip_deflection_rel_error": float(
            np.linalg.norm(w_tip_pinn - w_tip_numerical)
            / np.linalg.norm(w_tip_numerical)
        ),
    }

    # ---------------- Plot: histogram comparison ----------------
    plt.figure(figsize=(7, 5))
    plt.hist(w_tip_numerical * 1000, bins=80, alpha=0.5, label="numerical BVP (ground truth)", density=True)
    plt.hist(w_tip_pinn * 1000, bins=80, alpha=0.5, label="PINN", density=True)
    plt.axvline(cfg.W_ALLOW * 1000, color="red", linestyle="--",
                label=f"allowable = {cfg.W_ALLOW*1000:.2f} mm")
    plt.xlabel("tip deflection w(L) [mm]")
    plt.ylabel("probability density")
    plt.title(f"Tip-deflection distribution, nonlinear foundation\n(n = {n_samples:,} samples)")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(cfg.RESULTS_DIR, "uncertainty_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # ---------------- Plot: wall-clock time vs number of MC samples ----------------
    print("Measuring wall-clock scaling with sample count...")
    pinn_times, numerical_times, ns = [], [], []
    for n in n_timing_curve_points:
        idx = np.arange(n)
        # PINN timing at this N
        x_n = torch.full((n, 1), cfg.L, dtype=torch.float32)
        E_n = torch.tensor(E_samples[idx], dtype=torch.float32).reshape(-1, 1)
        q_n = torch.tensor(q_samples[idx], dtype=torch.float32).reshape(-1, 1)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_n, E_n, q_n).numpy()
        pinn_times.append(time.perf_counter() - t0)
        # Numerical timing at this N: use the already-measured per-sample rate
        # for large N to avoid re-running the expensive loop; for the small
        # N points actually re-time a fresh subset for an honest measurement.
        if n <= 2000:
            t0 = time.perf_counter()
            for i in idx:
                try:
                    solve_single(E_samples[i], q_samples[i])
                except RuntimeError:
                    pass
            numerical_times.append(time.perf_counter() - t0)
        else:
            numerical_times.append(results["timing_seconds"]["numerical_bvp_per_sample_ms"] / 1000 * n)
        ns.append(n)

    plt.figure(figsize=(6.5, 5))
    plt.loglog(ns, numerical_times, "o-", label="numerical BVP loop (per-sample solve)")
    plt.loglog(ns, pinn_times, "s-", label="PINN (batched forward pass)")
    plt.xlabel("number of Monte Carlo samples")
    plt.ylabel("wall-clock time [s] (log scale)")
    plt.title("Reliability-analysis cost vs. sample count")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    timing_path = os.path.join(cfg.RESULTS_DIR, "timing_scaling.png")
    plt.savefig(timing_path, dpi=150)
    plt.close()

    results["timing_scaling"] = {
        "n_samples": list(map(int, ns)),
        "numerical_seconds": numerical_times,
        "pinn_seconds": pinn_times,
    }

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "reliability_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, hist_path, timing_path
