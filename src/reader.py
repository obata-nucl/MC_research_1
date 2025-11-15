import csv
import numpy as np

from src.utils import load_config, _parse_pattern_name

CONFIG = load_config()

def load_eval_results(top_k: int = 5) -> list[list[int]]:
    """ load top-k evaluation reults from eval_summary.csv """
    load_dir = CONFIG["paths"]["results_dir"] / "evaluation"
    summary_path = load_dir / "eval_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"eval_summary.csv not found: {summary_path}")
    patterns = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("pattern", "")
            try:
                rmse = float(r.get("total_RMSE", "inf"))
            except ValueError:
                continue
            if not name or not np.isfinite(rmse):
                continue
            patterns.append(_parse_pattern_name(name))
            if len(patterns) >= top_k:
                break
    return patterns