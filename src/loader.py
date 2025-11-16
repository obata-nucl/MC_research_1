import csv
import numpy as np

from src.utils import load_config, _parse_pattern_name

CONFIG = load_config()

def load_eval_summary(top_k: int = 5) -> tuple[list[list[int]], list[list[int]]]:
    """ load top-k evaluation reults from eval_summary.csv """
    load_dir = CONFIG["paths"]["results_dir"] / "evaluation"
    summary_path = load_dir / "eval_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"eval_summary.csv not found: {summary_path}")
    rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("pattern", "")
            try:
                energy_RMSE = float(r.get("energy_RMSE", "inf"))
                ratio_RMSE = float(r.get("ratio_RMSE", "inf"))
            except ValueError:
                continue
            if not (np.isfinite(energy_RMSE) and np.isfinite(ratio_RMSE)):
                continue
            rows.append({"name": name, "energy": energy_RMSE, "ratio": ratio_RMSE, "total": energy_RMSE + ratio_RMSE})
    rows_by_ratio = sorted(rows, key=lambda x: x["ratio"])[:top_k]
    rows_by_total = sorted(rows, key=lambda x: x["total"])[:top_k]
    patterns_by_ratio = [_parse_pattern_name(r["name"]) for r in rows_by_ratio]
    patterns_by_total = [_parse_pattern_name(r["name"]) for r in rows_by_total]
    return patterns_by_ratio, patterns_by_total

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