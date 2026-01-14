from __future__ import annotations
import torch

def IBM2_PES(params: torch.Tensor, n_pi: torch.Tensor, n_nu: torch.Tensor, beta_f: torch.Tensor) -> torch.Tensor:
    """ Compute IBM2 potential energy surface (PES) """
    eps = params[:, 0].unsqueeze(1)
    kappa = params[:, 1].unsqueeze(1)
    chi_pi = params[:, 2].unsqueeze(1)
    chi_nu = params[:, 3].unsqueeze(1)
    beta_b = beta_f*4.0

    beta2 = beta_b**2
    deno1 = 1 + beta2
    deno2 = deno1**2
    SQRT_2_7 = 0.5345224838248488

    term_1 = eps*(n_pi + n_nu)*beta2/deno1
    term_2 = n_pi*n_nu*kappa*beta2*(
        4.0 - 2.0*SQRT_2_7*(chi_nu + chi_pi)*beta_b + 2.0*chi_nu*chi_pi*beta2/7.0
    )/deno2

    return term_1 + term_2
