import matplotlib.pyplot as plt
import numpy as np

from src.data import load_raw_expt_spectra
from src.loader import load_eval_results
from src.utils import load_config

CONFIG = load_config()

def plot_spectra(pred_spectra: np.ndarray, expt_spectra: np.ndarray, level_labels: list[str] = ["2+_1", "4+_1", "6+_1", "0+_2"], markers: list[str] = ['o', 's', '^', 'D']) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].set_title("Theory", fontsize=16)
    ax[1].set_title("Expt.", fontsize=16)
    for a in ax:
        a.set_ylim(0, 2.5)
        a.set_xlabel("Neutron Number", fontsize=14)
        a.set_ylabel("Energy [MeV]", fontsize=14)
        a.tick_params(axis="both", which="major", labelsize=12)
    for i in range(len(level_labels)):
        ax[0].plot(pred_spectra[:, 0].astype(int), pred_spectra[:, i + 1], marker=markers[i], label=level_labels[i])
        ax[1].plot(expt_spectra[:, 0].astype(int), expt_spectra[:, i + 1], marker=markers[i], label=level_labels[i])
    for a in ax:
        a.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig