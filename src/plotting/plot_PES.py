import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from src.physics import IBM2_PES
from src.utils import load_config

CONFIG = load_config()

def calc_PES(params: np.ndarray, n_pi: int, n_nu: int, beta_f_arr: np.ndarray) -> np.ndarray:
    """ calculate PES for one nucleus with given N """
    params = np.asarray(params)
    beta_f_arr = np.asarray(beta_f_arr)

    # Convert to tensors and shape for single set calculation
    params_tensor = torch.from_numpy(params.astype(np.float32)).unsqueeze(0)  # (1, 3)
    beta_f_tensor = torch.from_numpy(beta_f_arr.astype(np.float32)).unsqueeze(0)  # (1, num_beta)
    # Convert n_pi and n_nu to tensors for consistent arithmetic and broadcasting
    n_pi_tensor = beta_f_tensor.new_full(beta_f_tensor.shape, float(n_pi))
    n_nu_tensor = beta_f_tensor.new_full(beta_f_tensor.shape, float(n_nu))

    with torch.no_grad():
        pes_tensor = IBM2_PES(params_tensor, n_pi_tensor, n_nu_tensor, beta_f_tensor)

    # pes_tensor shape is (1, nbeta) -> return 1D numpy array
    return pes_tensor.squeeze(0).numpy()

def plot_n_PES(ax: plt.Axes, N: int, n_pi: int, n_nu: int, beta_f_arr: np.ndarray, params: np.ndarray, expt_PES: dict[tuple[int, int], np.ndarray]) -> plt.Axes:
    """ plot PES of one nucleus with given N """
    pred_PES = calc_PES(params, 6, n_nu, beta_f_arr)
    ax.plot(beta_f_arr, pred_PES, linestyle='-', color="black", label="IBM PES")

    min_idx = np.argmin(pred_PES)
    ax.plot(beta_f_arr[min_idx], pred_PES[min_idx], marker='ro', markersize=6)
    ax.plot(expt_PES[:, 0])

def main():
    beta_f_arr = np.arange(-0.45, 0.61, 0.01)

if __name__ == "__main__":
    main()