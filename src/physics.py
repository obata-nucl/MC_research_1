from __future__ import annotations
import math
import torch

def IBM2_PES(params: torch.Tensor, n_pi: torch.Tensor, n_nu: torch.Tensor, beta_f: torch.Tensor) -> torch.Tensor:
    """ Compute IBM2 potential energy surface (PES) """
    eps = params[:, 0].unsqueeze(1)
    kappa = params[:, 1].unsqueeze(1)
    chi_n = params[:, 2].unsqueeze(1)
    beta_b = beta_f*5.0

    beta2 = beta_b**2
    A = 1/(1 + beta2)
    SQRT_2_OVER_7 = torch.tensor(math.sqrt(2.0/7.0), dtype=params.dtype, device=params.device)

    term_1 = eps*(n_pi + n_nu)*beta2*A
    term_2 = n_pi*n_nu*kappa*beta2*A*A*(
        4.0 - 2.0*SQRT_2_OVER_7*(chi_n - 0.5)*beta_b - chi_n*beta2/7.0
    )

    return term_1 + term_2
