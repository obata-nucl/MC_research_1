import matplotlib.pyplot as plt
import numpy as np

from src.data import load_raw_expt_spectra
from src.loader import load_eval_results
from src.utils import load_config

CONFIG = load_config()

def plot_spectra(pred_data: np.ndarray, expt_data: np.ndarray, level_labels: list[str] = ["2+_1", "4+_1", "6+_1", "0+_2"], markers: list[str] = ['o', 's', '^', 'D']) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].set_title("Theory", fontsize=16)
    ax[1].set_title("Expt.", fontsize=16)
    for a in ax:
        a.set_ylim(0, 2.5)
        a.set_xlabel("Neutron Number", fontsize=14)
        a.set_ylabel("Energy [MeV]", fontsize=14)
        a.tick_params(axis="both", which="major", labelsize=12)
    for i in range(len(level_labels)):
        ax[0].plot(pred_data[:, 0].astype(int), pred_data[:, i + 1], marker=markers[i], label=level_labels[i])
        ax[1].plot(expt_data[:, 0].astype(int), expt_data[:, i + 1], marker=markers[i], label=level_labels[i])
    for a in ax:
        a.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig

def plot_ratio(pred_data: np.ndarray, expt_data: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pred_data[:, 0].astype(int), pred_data[:, 5], marker='D', color="#2A23F3", linewidth=2.0, label="Theory Ratio")
    ax.plot(expt_data[:, 0].astype(int), expt_data[:, 5], marker='D', color="#5C006EFF", linestyle="--", linewidth=1.8, label="Expt. Ratio")
    ax.set_title("E(4+)/E(2+) Ratio", fontsize=16)
    ax.set_ylim(1.0, 4.0)
    ax.set_xlabel("Neutron Number", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig

def plot_params(pred_data: np.ndarray, labels: dict[str, str] = {"eps": r"$\varepsilon$", "kappa": r"$\kappa$", "chi_n": r"$\chi_\nu$"}, lims: dict[str, tuple[float, float]] = {"eps": [0, 1.5], "kappa": [-0.5, 0], "chi_n": [-2.0, 0]}) -> plt.Figure:
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 5))

    for i, param_name in enumerate(labels.keys()):
        ax = axes[i]
        ax.plot(pred_data[:, 0].astype(int), pred_data[:, i + 6], linestyle='-', color="black", marker='o')
        ax.set_ylim(lims[param_name])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    return fig

def save_fig(fig: plt.Figure, pattern_name: str, filename: str) -> None:
    save_dir = CONFIG["paths"]["results_dir"] / "images" / pattern_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / filename
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_stem}.pdf", bbox_inches='tight')
    return