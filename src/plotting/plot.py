from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from src.data import load_raw_expt_spectra
from src.loader import load_eval_results
from src.utils import load_config

CONFIG = load_config()

# eval_results = ["N", "E2+_1", "E4+_1", "E6+_1", "E0+_2", "R_4/2", "eps", "kappa", "chi_n"]
# expt_spectra = {(p, n), np.ndarray}

def _plot_spectra(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray], level_labels: list[str] = ["2+_1", "4+_1", "6+_1", "0+_2"], markers: list[str] = ['o', 's', '^', 'D']) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    expt_keys = list(expt_data.keys())
    expt_energies = np.array([expt_data[key] for key in expt_keys])
    ax[0].set_title("Theory", fontsize=16)
    ax[1].set_title("Expt.", fontsize=16)
    for a in ax:
        a.set_ylim(0, 2.5)
        a.set_xlabel("Neutron Number", fontsize=14)
        a.set_ylabel("Energy [MeV]", fontsize=14)
        a.tick_params(axis="both", which="major", labelsize=12)
    for i in range(len(level_labels)):
        ax[0].plot(pred_data[:, 0].astype(int), pred_data[:, i + 1], marker=markers[i], label=level_labels[i])
        ax[1].plot([key[1] for key in expt_keys], expt_energies[:, i], marker=markers[i], label=level_labels[i])
    for a in ax:
        a.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig

def _plot_ratio(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    expt_keys = list(expt_data.keys())
    expt_ratios = np.array([expt_data[key][4] for key in expt_keys])
    ax.plot(pred_data[:, 0].astype(int), pred_data[:, 5], marker='D', color="#2A23F3", linewidth=2.0, label="Theory Ratio")
    ax.plot([key[1] for key in expt_keys], expt_ratios, marker='D', color="#5C006EFF", linestyle="--", linewidth=1.8, label="Expt. Ratio")
    ax.set_title("E(4+)/E(2+) Ratio", fontsize=16)
    ax.set_ylim(1.0, 4.0)
    ax.set_xlabel("Neutron Number", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()
    return fig

def _plot_params(pred_data: np.ndarray, labels: dict[str, str] = {"eps": r"$\varepsilon$", "kappa": r"$\kappa$", "chi_n": r"$\chi_\nu$"}, lims: dict[str, tuple[float, float]] = {"eps": [0, 1.5], "kappa": [-0.5, 0], "chi_n": [-2.0, 0]}) -> plt.Figure:
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 5))

    for i, param_name in enumerate(labels.keys()):
        ax = axes[i]
        ax.plot(pred_data[:, 0].astype(int), pred_data[:, i + 6], linestyle='-', color="black", marker='o')
        ax.set_title(labels[param_name], fontsize=20)
        ax.set_ylim(lims[param_name])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    return fig

def save_fig(fig: plt.Figure, filename: str, save_dir: Path = None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / filename
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_stem}.pdf", bbox_inches='tight')
    plt.close(fig)
    return

def main():
    expt_data = load_raw_expt_spectra(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"]
    )
    for pattern_name, pred_data in load_eval_results().items():
        fig_spectra = _plot_spectra(pred_data, expt_data)
        save_fig(fig_spectra, "spectra", CONFIG["paths"]["results_dir"] / "images" / pattern_name)

        fig_ratio = _plot_ratio(pred_data, expt_data)
        save_fig(fig_ratio, "ratio", CONFIG["paths"]["results_dir"] / "images" / pattern_name)

        fig_params = _plot_params(pred_data)
        save_fig(fig_params, "params", CONFIG["paths"]["results_dir"] / "images" / pattern_name)
    return

if __name__ == "__main__":
    main()
