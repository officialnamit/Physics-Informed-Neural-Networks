"""
Physics-informed loss for the nonlinear beam-on-foundation problem.

The only change from the plain cantilever's loss is the PDE residual itself,
which now includes the linear and cubic foundation terms:
    E*I*w'''' + K1*w + K3*w^3 - q = 0
Everything else (automatic differentiation for the derivatives, the boundary
loss, the overall training recipe) is identical to the plain-beam case.
"""
import torch

from . import config as cfg


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def pde_residual_loss(model, x, E, q):
    w = model(x, E, q)
    w_x = _grad(w, x)
    w_xx = _grad(w_x, x)
    w_xxx = _grad(w_xx, x)
    w_xxxx = _grad(w_xxx, x)

    residual = E * cfg.I * w_xxxx + cfg.K1 * w + cfg.K3 * w**3 - q
    norm = cfg.Q_NOM
    return torch.mean((residual / norm) ** 2)


def boundary_loss(model, fixed_pt, free_pt):
    x0, E0, q0 = fixed_pt
    w0 = model(x0, E0, q0)
    w0_x = _grad(w0, x0)

    xL, EL, qL = free_pt
    wL = model(xL, EL, qL)
    wL_x = _grad(wL, xL)
    wL_xx = _grad(wL_x, xL)
    wL_xxx = _grad(wL_xx, xL)

    loss_w0 = torch.mean((w0 / cfg.W_SCALE) ** 2)
    loss_w0x = torch.mean((w0_x * cfg.L / cfg.W_SCALE) ** 2)
    loss_M_L = torch.mean((EL * cfg.I * wL_xx / cfg.Q_NOM / cfg.L**2) ** 2)
    loss_V_L = torch.mean((EL * cfg.I * wL_xxx / cfg.Q_NOM / cfg.L) ** 2)

    return loss_w0 + loss_w0x + loss_M_L + loss_V_L


def total_loss(model, colloc, fixed_pt, free_pt):
    x, E, q = colloc
    l_pde = pde_residual_loss(model, x, E, q)
    l_bc = boundary_loss(model, fixed_pt, free_pt)
    return l_pde + cfg.BC_WEIGHT * l_bc, l_pde.item(), l_bc.item()
