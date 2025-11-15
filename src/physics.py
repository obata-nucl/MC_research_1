import math
import torch

def IBM2_PES(params: torch.Tensor, n_pi: torch.Tensor, n_nu: torch.Tensor, beta_f: torch.Tensor) -> torch.Tensor:
    """ Compute IBM2 potential energy surface (PES) """
    beta_b = beta_f*5.0

    beta2 = beta_b**2
    A = 1/(1 + beta2)
    SQRT_2_OVER_7 = math.sqrt(2.0/7.0)

    term_1 = params[:, 0]*(n_pi + n_nu)*beta2*A
    term_2 = n_pi*n_nu*params[:, 1]*beta2*A*A*(
        4 - 2*SQRT_2_OVER_7*(params[:, 2] - 0.5)*beta_b - params[:, 2]*beta2/7
    )

    return term_1 + term_2