"""
Training loop: standard optimization formulation (minimize a loss functional
over network parameters theta) using Adam, a stochastic-gradient-descent
variant. Fresh collocation/boundary points are re-sampled periodically so the
network doesn't overfit to one fixed point cloud (a form of stochasticity in
addition to mini-batch noise).
"""
import json
import os
import time

import torch

from . import config as cfg
from .data_generation import sample_boundary_points, sample_collocation_points
from .losses import total_loss
from .pinn_model import BeamPINN


def train():
    torch.manual_seed(cfg.SEED)
    model = BeamPINN().to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history = {"epoch": [], "total": [], "pde": [], "bc": []}

    t0 = time.time()
    for epoch in range(1, cfg.EPOCHS + 1):
        # Resample the point cloud every 200 epochs (fresh "stochastic" data)
        if epoch == 1 or epoch % 200 == 0:
            colloc = sample_collocation_points()
            fixed_pt, free_pt = sample_boundary_points()

        optimizer.zero_grad()
        loss, l_pde, l_bc = total_loss(model, colloc, fixed_pt, free_pt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.PRINT_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | total {loss.item():.6e} | "
                  f"pde {l_pde:.6e} | bc {l_bc:.6e}")
        history["epoch"].append(epoch)
        history["total"].append(loss.item())
        history["pde"].append(l_pde)
        history["bc"].append(l_bc)

    adam_time = time.time() - t0
    print(f"Adam stage complete in {adam_time:.1f} s")

    # --- Stage 2: L-BFGS fine-tuning ---------------------------------------
    # A standard PINN trick: Adam gets the loss into a good basin quickly,
    # then a quasi-Newton method (L-BFGS) squeezes out the remaining error
    # far more precisely than continued first-order SGD-style steps would.
    print("Starting L-BFGS fine-tuning stage...")
    colloc = sample_collocation_points()
    fixed_pt, free_pt = sample_boundary_points()
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=cfg.LBFGS_ITERS,
        history_size=50, tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        loss, l_pde, l_bc = total_loss(model, colloc, fixed_pt, free_pt)
        loss.backward()
        closure.last = (loss.item(), l_pde, l_bc)
        return loss

    closure.last = (None, None, None)
    t1 = time.time()
    lbfgs.step(closure)
    lbfgs_time = time.time() - t1
    l_total, l_pde, l_bc = closure.last
    print(f"L-BFGS stage complete in {lbfgs_time:.1f} s | "
          f"final total {l_total:.6e} | pde {l_pde:.6e} | bc {l_bc:.6e}")
    history["epoch"].append(cfg.EPOCHS + cfg.LBFGS_ITERS)
    history["total"].append(l_total)
    history["pde"].append(l_pde)
    history["bc"].append(l_bc)

    train_time = adam_time + lbfgs_time
    print(f"Training complete in {train_time:.1f} s total "
          f"(Adam {adam_time:.1f}s + L-BFGS {lbfgs_time:.1f}s)")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.RESULTS_DIR, "pinn_model.pt"))
    with open(os.path.join(cfg.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model, history, train_time


if __name__ == "__main__":
    train()
