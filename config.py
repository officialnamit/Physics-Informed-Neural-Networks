"""
Configuration for Project 5, extended version: a cantilever beam resting on a
NONLINEAR elastic foundation. Unlike the plain cantilever beam (which has a
closed-form solution), this problem has NO closed-form solution -- the only
way to get a "ground truth" answer is to numerically solve a nonlinear
boundary-value problem (BVP) for every single material/load combination.
That per-sample cost is what makes the PINN's speed advantage genuinely
matter, rather than being a demonstration on a toy problem that already had a
free, instant, exact formula.
"""
import torch

# ---------------------------------------------------------------------------
# Physical problem: cantilever beam, fixed at x=0, free at x=L, resting along
# its whole length on an elastic foundation with a hardening cubic term, under
# a uniformly distributed transverse load q.
#
# Governing ODE:   E*I*w''''(x) + k1*w(x) + k3*w(x)^3 = q
# Boundary conditions (same as the plain cantilever):
#   w(0)=0, w'(0)=0                  (fixed end)
#   w''(L)=0, w'''(L)=0              (free end: zero moment, zero shear)
#
# The k3*w^3 term makes this a NONLINEAR ODE: there is no algebraic formula
# for w(x) anymore, even though q is constant. Every "exact" answer requires
# numerically solving the ODE.
# ---------------------------------------------------------------------------

L = 1.0                  # beam length [m]
I = 2.0e-7                # second moment of area [m^4]
K1 = 2000.0               # linear foundation modulus [N/m/m]
K3 = 2.0e9                # cubic (hardening) foundation modulus [N/m^4]

# Random / uncertain material & load properties (same distributions as the
# linear cantilever study, so results are directly comparable)
E_NOM = 200.0e9
E_COV = 0.10
Q_NOM = 1000.0
Q_COV = 0.15

W_ALLOW = L / 250.0       # allowable tip deflection [m]

W_SCALE = Q_NOM * L**4 / (24.0 * E_NOM * I)   # output normalization for the network

# ---------------------------------------------------------------------------
# Training hyperparameters (same architecture family as the linear study)
# ---------------------------------------------------------------------------
N_COLLOCATION = 1200
N_BOUNDARY = 150
HIDDEN_LAYERS = 4
HIDDEN_WIDTH = 40
LR = 2e-3
EPOCHS = 3500
BC_WEIGHT = 10.0
PRINT_EVERY = 350
LBFGS_ITERS = 400
SEED = 42

# ---------------------------------------------------------------------------
# Monte Carlo / uncertainty propagation & the numerical-BVP baseline
# ---------------------------------------------------------------------------
N_MC_SAMPLES = 20000
BVP_TOL = 1e-6
BVP_MESH_POINTS = 30
BVP_MAX_NODES = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
