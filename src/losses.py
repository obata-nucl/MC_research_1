from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from src.physics import IBM2_PES

def loss_fn(params: torch.Tensor, n_pi: torch.Tensor, n_nu: torch.Tensor, beta_f: torch.Tensor, HFB_energies: torch.Tensor) -> torch.Tensor:
    """ Compute MSE loss between predicted IBM2 energies and target HFB energies """
    pred_energies = IBM2_PES(params, n_pi, n_nu, beta_f)
    return nn.MSELoss()(pred_energies, HFB_energies)

def calc_sse(pred_energies: torch.Tensor, target_energies: torch.Tensor) -> tuple[float, int]:
    """ calculate sum of squared errors """
    sse = 0.0
    count = 0
    for pred, target in zip(pred_energies, target_energies):
        if pred is None or target is None:
            continue
        try:
            pred_f = float(pred)
            target_f = float(target)
        except (TypeError, ValueError):
            continue
        if np.isnan(pred_f) or np.isnan(target_f):
            continue
        sse += (pred_f - target_f) * (pred_f - target_f)
        count += 1
    return sse, count
