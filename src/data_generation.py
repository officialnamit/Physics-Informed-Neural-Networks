"""
Programmatic training-data generation.

No external dataset is used. Instead we generate:
  1. Collocation points: (x, E, q) triples sampled across the physical
     domain x in [0, L] and across plausible ranges of the uncertain
     material property E and load q. The PDE residual is enforced here.
  2. Boundary points: x fixed at 0 or L, paired with sampled (E, q), where
     the exact boundary conditions are enforced.

Sampling (x, E, q) jointly -- rather than only x -- is what turns this into
a *parametric* PINN: the network learns the whole family of solutions
w(x; E, q), which is exactly what the later uncertainty-propagation step
needs (fast forward passes at new E, q without retraining).
"""
import torch

from . import config as cfg


def _sample_E(n, device):
    # Lognormal with the specified coefficient of variation, mean = E_NOM
    sigma = (torch.log(torch.tensor(1.0 + cfg.E_COV**2))).sqrt()
    mu = torch.log(torch.tensor(cfg.E_NOM)) - 0.5 * sigma**2
    z = torch.randn(n, 1, device=device)
    return torch.exp(mu + sigma * z)


def _sample_q(n, device):
    # Normal with specified CoV, clipped to stay positive (physical load)
    q = cfg.Q_NOM + cfg.Q_COV * cfg.Q_NOM * torch.randn(n, 1, device=device)
    return torch.clamp(q, min=0.05 * cfg.Q_NOM)


def sample_collocation_points(n=cfg.N_COLLOCATION, device=cfg.DEVICE):
    x = torch.rand(n, 1, device=device) * cfg.L
    E = _sample_E(n, device)
    q = _sample_q(n, device)
    x.requires_grad_(True)
    return x, E, q


def sample_boundary_points(n=cfg.N_BOUNDARY, device=cfg.DEVICE):
    """Returns fixed-end and free-end point sets, each paired with sampled (E,q)."""
    x0 = torch.zeros(n, 1, device=device, requires_grad=True)
    xL = torch.full((n, 1), cfg.L, device=device, requires_grad=True)
    E0 = _sample_E(n, device)
    q0 = _sample_q(n, device)
    EL = _sample_E(n, device)
    qL = _sample_q(n, device)
    return (x0, E0, q0), (xL, EL, qL)
