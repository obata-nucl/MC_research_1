import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from src.data import get_n_nu, load_raw_HFB_energies
from src.loader import load_eval_results
from src.physics import IBM2_PES
from src.plotting.plot import save_fig
from src.utils import load_config

CONFIG = load_config()

def _calc_PES(params: np.ndarray, n_pi: int, n_nu: int, beta_f_arr: np.ndarray) -> np.ndarray:
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

def _plot_n_PES(ax: plt.Axes, P:int, N: int, n_pi: int, n_nu: int, beta_f_arr: np.ndarray, params: np.ndarray, expt_PES: np.ndarray) -> plt.Axes:
    """ plot PES of one nucleus with given N """
    pred_PES = _calc_PES(params, 6, n_nu, beta_f_arr)
    ax.plot(beta_f_arr, pred_PES, linestyle='-', color="black", label="IBM PES")

    idx_min_calc = np.argmin(pred_PES)
    ax.plot(beta_f_arr[idx_min_calc], pred_PES[idx_min_calc], 'ro', markersize=6)

    ax.plot(expt_PES[:, 0], expt_PES[:, 1], linestyle="--", color="tab:orange", label="HFB PES")
    idx_min_expt = np.argmin(expt_PES[:, 1])
    ax.plot(expt_PES[idx_min_expt, 0], expt_PES[idx_min_expt, 1], 'bo', markersize=6)
    mass_number = P + N
    ax.set_title(rf"$^{mass_number}\mathrm{{Sm}}$", fontsize=18)
    ax.set_xlabel(r"$\beta$", fontsize=14)
    ax.set_ylabel("Energy [MeV]", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="best", fontsize=12)
    return ax

def main():
    beta_f_arr = np.arange(-0.45, 0.61, 0.01)
    expt_PES = load_raw_HFB_energies(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"],
        CONFIG["nuclei"]["n_step"]
    )
    for pattern_name, pred_data in load_eval_results().items():
        Protons = 62    # for Sm isotopes
        Neutrons = pred_data[:, 0].astype(int)
        N_nu = [get_n_nu(N) for N in Neutrons]
        n_panels = len(Neutrons)
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
        base_w, base_h = 5.0, 4.0
        fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows), sharex=True, sharey=True)
        for i, (n, n_nu) in enumerate(zip(Neutrons, N_nu)):
            expt_PES_n = expt_PES.get((Protons, n))
            idx_beta0 = np.where(np.isclose(expt_PES_n[:, 0], 0.0))[0]
            if idx_beta0.size == 0:
                raise ValueError(f"No beta=0 point for N={n}")
            e0 = expt_PES_n[idx_beta0[0], 1]
            expt_PES_n[:, 1] -= e0
            ax = axes.ravel()[i]
            params = pred_data[i, 6:]
            _plot_n_PES(ax, Protons, n, 6, n_nu, beta_f_arr, params, expt_PES_n)
        fig.tight_layout()
        save_fig(fig, pattern_name, "PES")
    return

if __name__ == "__main__":
    main()