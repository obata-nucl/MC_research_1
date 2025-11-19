from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, _parse_pattern_name

CONFIG = load_config()

def load_eval_summary() -> pd.DataFrame:
    """ load top-k evaluation reults from eval_summary.csv """
    eval_summary_path = CONFIG["paths"]["results_dir"] / "evaluation" / "eval_summary.csv"
    df = pd.read_csv(eval_summary_path)
    return df

def load_eval_results() -> dict[str, np.ndarray]:
    """ load evaluation results(pattern, spectra, ratio, parameters) """
    load_dir = CONFIG["paths"]["results_dir"] / "evaluation"
    results = {}
    for file_name in load_dir.iterdir():
        if file_name.suffix == ".csv" and file_name.name != "eval_summary.csv":
            stem = file_name.stem
            data = np.loadtxt(file_name, delimiter=',', skiprows=1)
            results[stem] = data
    return results