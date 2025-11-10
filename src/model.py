import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int):
        super(NN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        eps = F.softplus(out[:, 0])
        kappa = -F.softplus(out[:, 1])
        chi_n = -F.softplus(out[:, 2])
        return torch.stack([eps, kappa, chi_n], dim=1)