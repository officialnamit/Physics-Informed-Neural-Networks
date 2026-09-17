"""
Project 5, extended: Fast Reliability Analysis for a beam on a NONLINEAR
elastic foundation -- a problem with no closed-form solution, so the timing
comparison between the PINN and the brute-force baseline is a genuine,
honest measurement rather than a demonstration against a free formula.

Run with:  python main.py
"""
import json
import os

from src import config as cfg
from src.evaluate import plot_loss_curve, validate_against_numerical
from src.train import train
from src.uncertainty import run_uncertainty_propagation


def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    summary = {}

    print("=" * 70)
    print("PHASE 1-3: Training the physics-informed neural network (PINN)")
    print("=" * 70)
    model, history, train_time = train()
    summary["training_time_seconds"] = train_time
    summary["final_losses"] = {
        "total": history["total"][-1],
        "pde": history["pde"][-1],
        "bc": history["bc"][-1],
    }

    print("\n" + "=" * 70)
    print("PHASE 3b: Validating PINN against the NUMERICAL ground truth")
    print("=" * 70)
    plot_loss_curve(history)
    errors, _ = validate_against_numerical(model)
    summary["validation_rel_l2_errors"] = errors
    for label, err in errors.items():
        print(f"  {label:28s} relative L2 error: {err:.4%}")

    print("\n" + "=" * 70)
    print("PHASE 4: Uncertainty propagation, reliability, and TIMING comparison")
    print("=" * 70)
    reliability, _, _ = run_uncertainty_propagation(model)
    summary["reliability"] = reliability
    print(json.dumps({k: v for k, v in reliability.items() if k != "timing_scaling"}, indent=2))

    with open(os.path.join(cfg.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll results, plots, and metrics saved to ./results/")


if __name__ == "__main__":
    main()
