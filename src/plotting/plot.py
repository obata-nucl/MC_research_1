from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from pathlib import Path
from src.data import load_raw_expt_spectra
from src.loader import load_eval_results
from src.utils import load_config

CONFIG = load_config()

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12

# eval_results = ["N", "E2+_1", "E4+_1", "E6+_1", "E0+_2", "R_4/2", "eps", "kappa", "chi_n"]
# expt_spectra = {(p, n), np.ndarray}

def _plot_spectra(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray], element_name: str = "", level_labels: list[str] = ["2+_1", "4+_1", "6+_1", "0+_2"], markers: list[str] = ['o', 's', '^', 'D']) -> tuple[plt.Figure, float]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    expt_keys = list(expt_data.keys())
    expt_energies = np.array([expt_data[key] for key in expt_keys])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    ax[0].set_title("Theory (IBM-2)", fontsize=16)
    ax[1].set_title("Experiment", fontsize=16)

    sort_idx = np.argsort(pred_data[:, 0].astype(int))
    sorted_pred = pred_data[sort_idx]
    neutron_numbers = sorted_pred[:, 0].astype(int)

    expt_neutrons = np.array([key[1] for key in expt_keys], dtype=int)
    expt_sort_idx = np.argsort(expt_neutrons)
    expt_neutrons = expt_neutrons[expt_sort_idx]
    expt_energies = expt_energies[expt_sort_idx]

    for i in range(len(level_labels)):
        color = colors[i % len(colors)]
        ax[0].plot(neutron_numbers, sorted_pred[:, i + 2], marker=markers[i], color=color, label=level_labels[i])
        ax[1].plot(expt_neutrons, expt_energies[:, i], marker=markers[i], color=color, label=level_labels[i])

    max_energy = 0.0
    for i in range(len(level_labels)):
        pred_series = sorted_pred[:, i + 2]
        expt_series = expt_energies[:, i]
        pred_max = np.nanmax(pred_series) if np.any(~np.isnan(pred_series)) else 0.0
        expt_max = np.nanmax(expt_series) if np.any(~np.isnan(expt_series)) else 0.0
        max_energy = max(max_energy, pred_max, expt_max)
    energy_limit = max(2.0, max_energy * 1.1)

    for a in ax:
        a.set_xlabel("Neutron Number", fontsize=14)
        a.set_ylabel("Energy [MeV]", fontsize=14)
        a.legend(loc="best", fontsize=12)
        a.grid(True, linestyle='--', alpha=0.5)
        a.tick_params(axis="both", which="major", labelsize=12)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_ylim(bottom=0.0)

    if element_name:
        fig.suptitle(element_name, fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig, energy_limit

def _plot_ratio(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray], element_name: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    expt_keys = list(expt_data.keys())
    expt_ratios = np.array([expt_data[key][4] for key in expt_keys])

    ax.plot(pred_data[:, 0].astype(int), pred_data[:, 6], marker='D', color="#2A23F3", linewidth=2.0, label="Theory Ratio")
    ax.plot([key[1] for key in expt_keys], expt_ratios, marker='D', color="#5C006E", linestyle="--", linewidth=1.8, label="Expt. Ratio")

    title = r"$E(4^+_1)/E(2^+_1)$ Ratio"
    if element_name:
        title = f"{element_name} {title}"
    ax.set_title(title, fontsize=16)

    ax.set_ylim(1.0, 3.5)
    ax.set_xlabel("Neutron Number", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=12)
    if element_name:
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig

def _plot_params(pred_data: np.ndarray, element_name: str = "", labels: dict[str, str] = {"eps": r"$\varepsilon$", "kappa": r"$\kappa$", "chi_pi": r"$\chi_\pi$", "chi_n": r"$\chi_\nu$"}, lims: dict[str, tuple[float, float]] = None) -> plt.Figure:
    keys = list(labels.keys())
    cols = 2
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    neutron_numbers = pred_data[:, 0].astype(int)

    for i, param_name in enumerate(keys):
        ax = axes_flat[i]
        values = pred_data[:, i + 7]
        label = labels[param_name]
        ax.plot(neutron_numbers, values, "o-", color="black", linewidth=1.8)
        ax.set_title(f"Evolution of {label}", fontsize=16)
        ax.set_xlabel("Neutron Number", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        if lims is not None and param_name in lims:
            ax.set_ylim(lims[param_name])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis('off')

    if element_name:
        fig.suptitle(f"{element_name} Parameters", fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig

def save_fig(fig: plt.Figure, filename: str, save_dir: Path = None, close_fig: bool = True) -> None:
    if save_dir is None:
        save_dir = Path(CONFIG["paths"]["results_dir"])
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / filename
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_stem}.pdf", bbox_inches='tight')
    if close_fig:
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
        # pred_data: [N, Z, E2, E4, E6, E0, R, eps, kappa, chi_n]
        # Split by Z
        unique_Zs = np.unique(pred_data[:, 1].astype(int))
        for z in unique_Zs:
            mask = (pred_data[:, 1].astype(int) == z)
            z_pred_data = pred_data[mask]
            
            # Filter expt data for this Z
            z_expt_data = {k: v for k, v in expt_data.items() if k[0] == z}
            
            if not z_expt_data:
                continue

            element_name = CONFIG["elements"].get(int(z), f"Z={z}")
            save_dir = CONFIG["paths"]["results_dir"] / "images" / pattern_name / str(z)

            fig_spectra, spectra_limit = _plot_spectra(z_pred_data, z_expt_data, element_name=element_name)
            save_fig(fig_spectra, "spectra", save_dir, close_fig=False)
            for axis in fig_spectra.axes:
                axis.set_ylim(0.0, spectra_limit)
            save_fig(fig_spectra, "spectra_common_scale", save_dir)

            g_level_labels = ["2+_1", "4+_1", "6+_1"]
            g_markers = ['o', 's', '^']
            fig_spectra_g, _ = _plot_spectra(
                z_pred_data,
                z_expt_data,
                element_name=element_name,
                level_labels=g_level_labels,
                markers=g_markers
            )
            save_fig(fig_spectra_g, "spectra_g", save_dir, close_fig=False)
            for axis in fig_spectra_g.axes:
                axis.set_ylim(0.0, spectra_limit)
            save_fig(fig_spectra_g, "spectra_g_common_scale", save_dir)

            fig_ratio = _plot_ratio(z_pred_data, z_expt_data, element_name=element_name)
            save_fig(fig_ratio, "ratio", save_dir)

            param_lims = {
                "eps": (0.0, 3.5),
                "kappa": (-1.0, 0.0),
                "chi_pi": (-2.0, 0),
                "chi_n": (-2.0, 0)
            }
            fig_params = _plot_params(z_pred_data, element_name=element_name, lims=param_lims)
            save_fig(fig_params, "params", save_dir)
    return

if __name__ == "__main__":
    main()
