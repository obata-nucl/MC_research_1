from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from src.utils import load_config

CONFIG = load_config()

def _load_loss_csv(csv_path: Path) -> np.ndarray:
    """ load the loss.csv file with numpy and return a 2D array """
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except OSError as exc:
        raise RuntimeError(f"Failed to read {csv_path}") from exc

    if data.size == 0:
        raise ValueError(f"No data rows found in {csv_path}")

    if data.ndim == 1:
        data = data[np.newaxis, :]

    expected_cols = 6
    if data.shape[1] < expected_cols:
        raise ValueError(f"Expected at least {expected_cols} columns in {csv_path}, got {data.shape[1]}")

    return data

def _plot_learning_curve(data: np.ndarray, title: str) -> plt.Figure:
    """ plot RMSE and learning rate over epochs

        data = [epoch, train_MSE, val_MSE, train_RMSE, val_RMSE, lr]
    """
    epochs = data[:, 0]
    train_rmse = data[:, 3]
    val_rmse = data[:, 4]
    lrs = data[:, 5]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title(f"Learning Curve — {title}", fontsize=18)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("RMSE", fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle=":")

    ax1.plot(epochs, train_rmse, label="train_RMSE", color="#1f77b4")
    ax1.plot(epochs, val_rmse, label="val_RMSE", color="#ff7f0e")
    ax1.legend(loc="best")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", fontsize=14)
    ax2.plot(epochs, lrs, color="#2ca02c", alpha=0.35, label="lr")
    ax2.set_yscale("log")
    fig.tight_layout()
    return fig



def main():
    results_dir = CONFIG["paths"]["results_dir"] / "training"

    pattern_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not pattern_dirs:
        print(f"No pattern directories under {results_dir}")
        return

    for pattern_dir in pattern_dirs:
        csv_path = pattern_dir / "loss.csv"
        if not csv_path.exists():
            print(f"Skipping {pattern_dir.name}: loss.csv not found")
            continue

        try:
            data = _load_loss_csv(csv_path)
        except Exception as e:
            print(f"Skipping {pattern_dir.name}: failed to load loss history: {e}")
            continue

        fig = _plot_learning_curve(data, pattern_dir.name)

        png_path = pattern_dir / "learning_curve.png"
        pdf_path = pattern_dir / "learning_curve.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved learning curves to {pattern_dir}")
    return

if __name__ == "__main__":
    main()
