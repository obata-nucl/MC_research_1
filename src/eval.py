from __future__ import annotations
import csv
import numpy as np
import os
import pandas as pd
import signal
import subprocess
import torch

from src.data import load_eval_dataset, load_raw_expt_spectra
from src.losses import calc_sse
from src.model import load_NN_model
from src.loader import load_eval_summary
from src.utils import load_config, get_all_patterns, _pattern_to_name, _parse_pattern_name

CONFIG = load_config()

Z_MAP = {
    50: "Sn", 52: "Te", 54: "Xe", 56: "Ba", 58: "Ce", 60: "Nd", 62: "Sm", 64: "Gd",
    66: "Dy", 68: "Er", 70: "Yb", 72: "Hf", 74: "W", 76: "Os", 78: "Pt", 80: "Hg", 82: "Pb"
}

def _run_npbos(command: list[str], timeout_sec: float=5.0) -> tuple[str, str, int]:
    """ run NPBOS programs and return IBM spectra as stdout, stderr, and return code """
    proc = None
    try:
        proc = subprocess.Popen(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        rc = proc.returncode
        return (stdout or "", stderr or "", int(rc))
    except subprocess.TimeoutExpired:
        try:
            if proc is not None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            if proc is not None:
                proc.communicate(timeout=timeout_sec)
        except Exception:
            try:
                if proc is not None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return ("", "timed out", -1)
    except Exception as e:
        try:
            if proc is not None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        return ("", str(e), -1)

def _evaluate_model(X_eval: torch.Tensor, X_eval_scaled: torch.Tensor, pattern: list[int], expt_spectra: dict[tuple[int, int], np.ndarray]) -> tuple[float, float]:
    """ evaluate each pattern model """
    model = load_NN_model(pattern)
    if model is None:
        raise ValueError("Model could not be loaded.")
    model.eval()

    num_eval_X = len(X_eval)
    energy_RMSE, ratio_RMSE = 0.0, 0.0
    energy_count_total, ratio_count = 0, 0

    for i in range(num_eval_X):
        n = int(X_eval[i, 0].item())
        p = int(X_eval[i, 1].item())
        n_nu = int(X_eval[i, 2].item())
        n_pi = int(X_eval[i, 3].item())
        element = Z_MAP.get(p, "Sm")
        expt_spectra_n = expt_spectra.get((p, n))

        if expt_spectra_n is None or expt_spectra_n.size == 0:
            # print(f"No expt data found for Z={p} N={n}")
            continue

        with torch.no_grad():
            outputs = model(X_eval_scaled[i].unsqueeze(0))
        
        pred_params = outputs.squeeze(0).numpy()
        sh_command = [
            "bash", CONFIG["paths"]["src_dir"] / "eval.sh",
            str(CONFIG["paths"]["NPBOS_dir"]),
            str(int(n + p)), str(int(n_nu)), str(int(n_pi)), element,
            *[f"{param:.3f}" for param in pred_params]
        ]

        stdout, stderr, rc = _run_npbos(sh_command)
        if rc != 0:
            print(f"timeout for N={n}, params = {pred_params}")
            continue
        try:
            pred_energies = list(map(float, stdout.strip().split()))
        except ValueError as e:
            print(f"Error parsing output: {stdout} - {e}")
            continue

        energy_sse, energy_count = calc_sse(pred_energies, expt_spectra_n)
        energy_RMSE += energy_sse
        energy_count_total += energy_count
        # ratio evaluation: require only first two levels to exist (no need for strict length equality)
        if len(pred_energies) == 4 and pred_energies[0] != 0 and expt_spectra_n[0] != 0:
            ratio_RMSE += ((pred_energies[1] / pred_energies[0]) - expt_spectra_n[4]) ** 2
            ratio_count += 1
    energy_RMSE = np.sqrt(energy_RMSE / energy_count_total) if energy_count_total > 0 else float('inf')
    ratio_RMSE = np.sqrt(ratio_RMSE / ratio_count) if ratio_count > 0 else float('inf')
    return energy_RMSE, ratio_RMSE

def _save_rmse_to_csv(patterns: list[list[int]], X_eval: torch.Tensor, X_eval_scaled: torch.Tensor, expt_spectra: dict[tuple[int, int], np.ndarray]) -> None:
    """ save RMSE results to .csv file """
    eval_summary = []
    calc_count = 0
    for pattern in patterns:
        try:
            energy_RMSE, ratio_RMSE = _evaluate_model(X_eval, X_eval_scaled, pattern, expt_spectra)
            eval_summary.append(
                {
                    "pattern": _pattern_to_name(pattern),
                    "energy_RMSE": energy_RMSE,
                    "ratio_RMSE": ratio_RMSE,
                }
            )
            calc_count += 1
            if calc_count % 100 == 0:
                print(f"Evaluated {calc_count} patterns")
        except Exception as e:
            print(f"Error evaluating pattern {pattern}: {e}")
            continue
    # Sort CSV lines by ratio_RMSE ascending as requested
    eval_summary.sort(key=lambda x: x["ratio_RMSE"])
    summary_dir = CONFIG["paths"]["results_dir"] / "evaluation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "eval_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pattern", "energy_RMSE", "ratio_RMSE"])
        for record in eval_summary:
            writer.writerow([
                record["pattern"],
                f"{record['energy_RMSE']:.6f}",
                f"{record['ratio_RMSE']:.6f}",
            ])
    return



def _save_spectra_to_csv(pattern: list[int], X_eval: torch.Tensor, X_eval_scaled: torch.Tensor, max_attempts: int = 5) -> None:
    """ save predicted spectra and parameters to .csv file """
    model = load_NN_model(pattern)
    result_dir = CONFIG["paths"]["results_dir"] / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / f"{_pattern_to_name(pattern)}.csv"
    import time
    header = ["N", "Z", "2+_1", "4+_1", "6+_1", "0+_2", "R_4/2", "eps", "kappa", "chi_pi", "chi_n"]
    attempt = 1
    while attempt <= max_attempts:
        rows = []
        failed = False
        for x_eval, x_eval_scaled in zip(X_eval, X_eval_scaled):
            n = int(x_eval[0].item())
            p = int(x_eval[1].item())
            n_nu = int(x_eval[2].item())
            n_pi = int(x_eval[3].item())
            element = Z_MAP.get(p, "Sm")
            with torch.no_grad():
                outputs = model(x_eval_scaled.unsqueeze(0))
            pred_params = outputs.squeeze(0).numpy()
            sh_command = [
                "bash", CONFIG["paths"]["src_dir"] / "eval.sh",
                str(CONFIG["paths"]["NPBOS_dir"]),
                str(int(n + p)), str(int(n_nu)), str(int(n_pi)), element,
                *[f"{param:.3f}" for param in pred_params]
            ]
            stdout, _, rc = _run_npbos(sh_command)
            if rc != 0:
                print(f"timeout for Z={p} N={n}, params = {pred_params} (attempt {attempt})")
                failed = True
                break
            try:
                pred_energies = list(map(float, stdout.strip().split()))
            except ValueError as e:
                print(f"Error parsing output for Z={p} N={n}: {stdout} - {e} (attempt {attempt})")
                failed = True
                break

            if len(pred_energies) == 4 and pred_energies[0] != 0:
                ratio = pred_energies[1] / pred_energies[0]
                rows.append([n, p, *pred_energies, f"{ratio:.3f}", *[f"{param:.3f}" for param in pred_params]])
        if not failed:
            # All N succeeded for this pattern -> write CSV and exit loop
            with open(save_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
            print(f"Saved spectra for pattern {_pattern_to_name(pattern)} (attempt {attempt})")
            break
        else:
            if attempt < max_attempts:
                print(f"Pattern {_pattern_to_name(pattern)}: attempt {attempt} failed; retrying...")
                attempt += 1
                time.sleep(min(1 * attempt, 3))
            else:
                print(f"Pattern {_pattern_to_name(pattern)}: failed after {max_attempts} attempts; skipping and continuing")
                return

def _sort_by(data: pd.DataFrame, key: str) -> pd.DataFrame:
    """ sort DataFrame by given key column """
    return data.sort_values(by=key)

def find_best_training_model(patterns: list[list[int]]) -> tuple[list[int], float]:
    """ find the pattern with the minimum validation RMSE from training logs """
    best_pattern = None
    min_val_rmse = float('inf')
    
    for pattern in patterns:
        pattern_name = _pattern_to_name(pattern)
        loss_path = CONFIG["paths"]["results_dir"] / "training" / pattern_name / "loss.csv"
        if not loss_path.exists():
            continue
        try:
            df = pd.read_csv(loss_path)
            if "val_RMSE" not in df.columns:
                continue
            # Get the minimum val_RMSE recorded during training
            val_rmse = df["val_RMSE"].min()
            if val_rmse < min_val_rmse:
                min_val_rmse = val_rmse
                best_pattern = pattern
        except Exception:
            continue
            
    return best_pattern, min_val_rmse

def main():
    X_eval, X_eval_scaled = load_eval_dataset("eval_dataset")

    expt_spectra = load_raw_expt_spectra(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"],
    )
    patterns = get_all_patterns(CONFIG["nn"]["nodes_options"], CONFIG["nn"]["layers_options"])
    
    print("Evaluating all models...")
    _save_rmse_to_csv(patterns, X_eval, X_eval_scaled, expt_spectra)

    eval_summary = load_eval_summary()
    
    # NPBOS best 2 models (Energy RMSE & Ratio RMSE)
    best_energy_row = eval_summary.loc[eval_summary["energy_RMSE"].idxmin()]
    best_ratio_row = eval_summary.loc[eval_summary["ratio_RMSE"].idxmin()]

    # Best total RMSE (Energy + Ratio)
    # Note: Depending on the scale of energy vs ratio, you might want to normalize them or use a weighted sum.
    # Here we simply sum them as requested.
    eval_summary["total_RMSE"] = eval_summary["energy_RMSE"] + eval_summary["ratio_RMSE"]
    best_total_row = eval_summary.loc[eval_summary["total_RMSE"].idxmin()]
    
    pattern_energy_best = _parse_pattern_name(best_energy_row["pattern"])
    pattern_ratio_best = _parse_pattern_name(best_ratio_row["pattern"])
    pattern_total_best = _parse_pattern_name(best_total_row["pattern"])
    
    print(f"Best NPBOS Energy RMSE: {best_energy_row['pattern']} (RMSE={best_energy_row['energy_RMSE']})")
    print(f"Best NPBOS Ratio RMSE: {best_ratio_row['pattern']} (RMSE={best_ratio_row['ratio_RMSE']})")
    print(f"Best NPBOS total RMSE: {best_total_row['pattern']} (RMSE={best_total_row['total_RMSE']})")

    # PES Training best model
    pattern_train_best, train_rmse = find_best_training_model(patterns)
    if pattern_train_best:
        print(f"Best PES Training RMSE: {_pattern_to_name(pattern_train_best)} (RMSE={train_rmse})")
    else:
        print("No training logs found to determine best PES model.")

    # Save spectra for these models
    target_patterns = []
    if pattern_train_best:
        target_patterns.append(pattern_train_best)
    target_patterns.append(pattern_energy_best)
    target_patterns.append(pattern_ratio_best)
    target_patterns.append(pattern_total_best)
    
    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in target_patterns:
        name = _pattern_to_name(p)
        if name not in seen:
            unique_patterns.append(p)
            seen.add(name)
            
    for pattern in unique_patterns:
        print(f"Saving spectra for pattern: {_pattern_to_name(pattern)}")
        _save_spectra_to_csv(pattern, X_eval, X_eval_scaled)

if __name__ == "__main__":
    main()
