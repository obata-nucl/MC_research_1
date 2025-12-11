from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pathlib import Path
from src.loader import load_eval_summary
from src.utils import load_config, _parse_pattern_name
from src.plotting.plot import save_fig

CONFIG = load_config()

def _patterns_to_matrix(patterns: pd.Series, max_layers: int = CONFIG["nn"]["layers_options"][-1]) -> np.ndarray:
    """ convert a series of pattern-name strings into an (N, max_layers) array 
        Shorter patterns are padded with np.nan 
        ex) 8-16-32-8 -> [8, 16, 32, 8, nan]
    """
    mat = np.full((len(patterns), max_layers), np.nan, dtype=float)
    for i, p in enumerate(patterns):
        nodes = _parse_pattern_name(p)
        for j, node in enumerate(nodes[:max_layers]):
            mat[i, j] = node
    return mat

def _plot_tendency(eval_summary: pd.DataFrame, metric: str = "total_RMSE") -> plt.Figure:
    """ 
    Plot metric distribution as a heatmap:
    X-axis: Total number of nodes in hidden layers
    Y-axis: Number of layers
    Color: Minimum metric value for that combination
    """
    df = eval_summary.copy()
    
    # Calculate total nodes and number of layers
    def get_specs(pattern_str):
        nodes = _parse_pattern_name(pattern_str)
        return sum(nodes), len(nodes)

    specs = df["pattern"].apply(get_specs)
    df["total_nodes"] = specs.apply(lambda x: x[0])
    df["num_layers"] = specs.apply(lambda x: x[1])
    
    # Aggregate min metric
    grouped = df.groupby(["total_nodes", "num_layers"])[metric].min().reset_index()
    
    # Create pivot table for heatmap
    # index=num_layers (Y), columns=total_nodes (X)
    pivot = grouped.pivot(index="num_layers", columns="total_nodes", values=metric)
    
    # Sort index and columns
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    
    # Fill gaps in X-axis (total_nodes) to ensure linear scale
    # We assume step is 8 based on node options [8, 16, 32, 64]
    step_nodes = 8
    min_nodes = pivot.columns.min()
    max_nodes = pivot.columns.max()
    
    all_nodes = np.arange(min_nodes, max_nodes + step_nodes, step_nodes)
    pivot = pivot.reindex(columns=all_nodes)
    
    # Fill gaps in Y-axis (num_layers)
    min_layers = pivot.index.min()
    max_layers = pivot.index.max()
    all_layers = np.arange(min_layers, max_layers + 1)
    pivot = pivot.reindex(index=all_layers)
    
    # Prepare meshgrid for pcolormesh
    # pcolormesh expects edges. 
    # X centers are all_nodes. Edges are all_nodes - step/2 and last + step/2
    x_edges = np.concatenate([all_nodes - step_nodes/2, [all_nodes[-1] + step_nodes/2]])
    y_edges = np.concatenate([all_layers - 0.5, [all_layers[-1] + 0.5]])
    
    # Z data (masked array to handle NaNs)
    Z = pivot.values
    Z_masked = np.ma.masked_invalid(Z)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot heatmap
    # vmin/vmax from actual data range (ignoring NaNs)
    vmin = grouped[metric].min()
    vmax = grouped[metric].max()
    
    mesh = ax.pcolormesh(x_edges, y_edges, Z_masked, cmap="viridis", 
                         vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.5)
    
    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"Minimum {metric}", fontsize=12)
    
    ax.set_title(f"Performance: Total Nodes vs Depth ({metric})", fontsize=16)
    ax.set_xlabel("Total Number of Nodes", fontsize=14)
    ax.set_ylabel("Number of Layers", fontsize=14)
    
    # Set ticks
    ax.set_yticks(all_layers)
    
    # Adjust X ticks to be readable
    if len(all_nodes) > 20:
        ax.set_xticks(all_nodes[::2])
    else:
        ax.set_xticks(all_nodes)
        
    ax.grid(True, linestyle=":", alpha=0.3)
    
    # Annotate the global best
    best_val = grouped[metric].min()
    best_rows = grouped[grouped[metric] == best_val]
    if not best_rows.empty:
        best_row = best_rows.iloc[0]
        ax.annotate(f"Best: {best_val:.3f}", 
                    xy=(best_row["total_nodes"], best_row["num_layers"]),
                    xytext=(20, 20), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                    color="red", fontweight="bold")

    fig.tight_layout()
    return fig

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tendency of NN architecture")
    parser.add_argument("--metric", type=str, default="sum_RMSE", help="Metric to sort and color by")
    args = parser.parse_args()

    eval_summary = load_eval_summary()
    if eval_summary is None or eval_summary.empty:
        print("No evaluation summary found.")
        return
    
    # Calculate sum of energy_RMSE and ratio_RMSE if requested
    if args.metric == "sum_RMSE":
        if "energy_RMSE" in eval_summary.columns and "ratio_RMSE" in eval_summary.columns:
            eval_summary["sum_RMSE"] = eval_summary["energy_RMSE"] + eval_summary["ratio_RMSE"]
        else:
            print("Columns 'energy_RMSE' and 'ratio_RMSE' are required for 'sum_RMSE'.")
            return

    if args.metric not in eval_summary.columns:
        print(f"Metric '{args.metric}' not found in summary. Available: {list(eval_summary.columns)}")
        return

    fig = _plot_tendency(eval_summary, metric=args.metric)
    
    save_dir = CONFIG["paths"]["results_dir"] / "images"
    save_fig(fig, "tendency_plot", save_dir)
    print(f"Saved tendency plot to {save_dir}")

if __name__ == "__main__":
    main()
