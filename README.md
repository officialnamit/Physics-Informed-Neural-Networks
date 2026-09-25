# Project 5 (extended): PINN vs. a Genuinely Expensive Baseline

This is the same idea as the original cantilever-beam project, but the beam
now rests on a **nonlinear elastic foundation** (a cubic hardening term).
That single change removes the closed-form shortcut: there is no algebraic
formula for the deflection anymore, so every "ground truth" evaluation
requires numerically solving a nonlinear boundary-value problem (BVP). That
makes the PINN-vs-baseline timing comparison a real, honest measurement
instead of a demonstration against a free formula.

## Headline result

| | Numerical BVP baseline | Trained PINN |
|---|---|---|
| Time for 20,000 Monte Carlo samples | **24.87 s** | **5.4 ms** |
| Per-sample cost | 1.24 ms | — (batched) |
| P(deflection > 4.00 mm) | 4.32% | 4.61% |

**~4,640× speedup**, with the reliability estimate still matching the
numerical ground truth closely.

## What's different from the original (linear) project

| | Linear cantilever (original) | Nonlinear foundation (this version) |
|---|---|---|
| Governing equation | `EI w'''' = q` | `EI w'''' + K1 w + K3 w^3 = q` |
| Ground truth | closed-form formula (instant) | `scipy.integrate.solve_bvp` (~1.2 ms/sample) |
| Baseline vs. PINN timing | baseline *faster* (both trivial) | PINN **~4,600x faster** |
| Everything else (network architecture, training recipe, data generation, uncertainty propagation) | — | unchanged |

## Files

```
Physics_Informed_Neural_Networks/
├── main.py
├── src/
│   ├── config.py              # physics constants incl. K1, K3 foundation terms
│   ├── numerical_baseline.py  # scipy solve_bvp -- the "expensive" ground truth
│   ├── pinn_model.py
│   ├── data_generation.py
│   ├── losses.py               # residual now includes K1*w + K3*w^3
│   ├── train.py
│   ├── evaluate.py             # validates against solve_bvp, not a formula
│   └── uncertainty.py          # times BOTH methods honestly, incl. a scaling sweep
└── results/
    ├── loss_curve.png
    ├── pinn_vs_numerical.png
    ├── uncertainty_histogram.png
    ├── timing_scaling.png        # NEW: wall-clock time vs. sample count, log-log
    ├── reliability_results.json
    ├── validation_errors.json
    ├── summary.json
    ├── training_history.json
    └── pinn_model.pt             # Model weights
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib scipy
python main.py
```

Expect roughly 2.5–3 minutes for training, plus about 25 seconds for the
20,000-sample numerical baseline loop (this wait is itself part of the point
being demonstrated).

## Why this baseline is a fair stand-in for "a real FE solve"

`solve_bvp` is a general-purpose nonlinear ODE solver: no problem-specific
shortcuts, no vectorization across samples, just "set up the equations, hand
them to a numerical solver, wait for convergence" -- exactly the workflow a
finite-element-based reliability study follows, just at a much smaller,
faster scale (1D beam instead of a 3D mesh). The qualitative conclusion --
that a trained PINN's forward-pass cost stays flat while the brute-force
cost grows linearly with the number of Monte Carlo samples -- carries over
directly to that more expensive, more realistic setting; only the absolute
numbers would change (from milliseconds-per-solve here to
seconds-or-minutes-per-solve there).
