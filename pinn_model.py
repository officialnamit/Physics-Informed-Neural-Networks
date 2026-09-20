"""
Same idea as the plain-cantilever PINN: w = f(x, E, q; theta), a feedforward
network that takes beam position AND the uncertain material/load properties
as inputs, so one trained model covers the whole family of solutions.
"""
import torch
import torch.nn as nn

from . import config as cfg


class BeamPINN(nn.Module):
    def __init__(self, hidden_layers=cfg.HIDDEN_LAYERS, hidden_width=cfg.HIDDEN_WIDTH):
        super().__init__()
        layers = [nn.Linear(3, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, E, q):
        x_hat = x / cfg.L
        E_hat = E / cfg.E_NOM
        q_hat = q / cfg.Q_NOM
        inp = torch.cat([x_hat, E_hat, q_hat], dim=1)
        w_hat = self.net(inp)
        return w_hat * cfg.W_SCALE
