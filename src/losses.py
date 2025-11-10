import torch
import torch.nn as nn
from physics import IBM2_PES

def loss_fn(params: torch.Tensor, n_pi: int, n_nu: torch.Tensor, beta_f: torch.Tensor, HFB_energies: torch.Tensor) -> torch.Tensor:
    """ Compute MSE loss between predicted IBM2 energies and target HFB energies """
    pred_energies = IBM2_PES(params, n_pi, n_nu, beta_f)
    return nn.MSELoss()(pred_energies, HFB_energies)