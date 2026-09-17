"""
Numerical "ground truth" solver for the nonlinear beam-on-foundation problem.

Because the governing ODE  E*I*w'''' + K1*w + K3*w^3 = q  is nonlinear, there
is no algebraic formula for w(x): every single (E, q) combination requires
solving a boundary-value problem (BVP) numerically. We use scipy's
`solve_bvp`, a standard, general-purpose, no-shortcuts nonlinear BVP solver ---
exactly the kind of tool a "brute-force" reliability study would have to call
once per Monte Carlo sample. This stands in for what would be a full nonlinear
finite-element solve in a more realistic 2D/3D structural problem.

We rewrite the 4th-order ODE as a system of four 1st-order ODEs:
    y1 = w
    y2 = w'
    y3 = w''
    y4 = w'''
    y4' = (q - K1*y1 - K3*y1^3) / (E*I)
"""
import numpy as np
from scipy.integrate import solve_bvp

from . import config as cfg


def _rhs(x, y, E, q):
    y1, y2, y3, y4 = y
    dy4 = (q - cfg.K1 * y1 - cfg.K3 * y1**3) / (E * cfg.I)
    return np.vstack([y2, y3, y4, dy4])


def _bc(ya, yb):
    # ya = state at x=0, yb = state at x=L
    return np.array([ya[0], ya[1], yb[2], yb[3]])


def solve_single(E, q, n_mesh=cfg.BVP_MESH_POINTS, tol=cfg.BVP_TOL, return_full=False):
    """
    Numerically solve the nonlinear cantilever-on-foundation BVP for one
    (E, q) combination. Returns the tip deflection w(L) by default, or the
    full (x, w(x)) arrays if return_full=True.
    """
    x = np.linspace(0.0, cfg.L, n_mesh)
    y0 = np.zeros((4, x.size))  # start from the "beam not yet bent" guess

    sol = solve_bvp(
        lambda x, y: _rhs(x, y, E, q),
        _bc,
        x,
        y0,
        tol=tol,
        max_nodes=cfg.BVP_MAX_NODES,
    )
    if sol.status != 0:
        raise RuntimeError(f"solve_bvp did not converge (status={sol.status}): {sol.message}")

    if return_full:
        x_fine = np.linspace(0.0, cfg.L, 200)
        w_fine = sol.sol(x_fine)[0]
        return x_fine, w_fine
    return sol.y[0, -1]
