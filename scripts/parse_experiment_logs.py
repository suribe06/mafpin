#!/usr/bin/env python3
"""Parse core-experiment logs into JSON (ponytail: one-off analysis helper)."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data/movielens/logs"


def parse_recommend_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {"log": path.name, "completed": "[RECOMMEND] Done." in text}
    if m := re.search(r"Command: (.+)", text):
        out["command"] = m.group(1).strip()
    if m := re.search(
        r"Baseline \(global test\) — RMSE: ([\d.]+)\s+MAE: ([\d.]+)\s+R²: ([\d.-]+)",
        text,
    ):
        out["baseline_global"] = {
            "rmse": float(m.group(1)),
            "mae": float(m.group(2)),
            "r2": float(m.group(3)),
        }
    if m := re.search(r"Best baseline params: (\{[^}]+\})\s+RMSE=([\d.]+)", text):
        out["baseline_cv"] = {"params": eval(m.group(1)), "rmse": float(m.group(2))}
    if m := re.search(r"Best enhanced params: (\{[^}]+\})\s+RMSE=([\d.]+)", text):
        out["enhanced_cv"] = {"params": eval(m.group(1)), "rmse": float(m.group(2))}
    if m := re.search(r"Social CMF best RMSE: ([\d.]+)", text):
        out["social_cv_rmse"] = float(m.group(1))
    if m := re.search(r"Social CMF best hyperparameters:\n(\{.*?\n\})", text, re.S):
        out["social_params"] = json.loads(m.group(1))
    if m := re.search(r"Enhanced CMF best hyperparameters:\n(\{.*?\n\})", text, re.S):
        out["enhanced_params"] = json.loads(m.group(1))
    out["network_best"] = {}
    for pat, key in [
        (
            r"Exponential\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)\s+improvement=([+-]?[\d.]+)%",
            "exponential",
        ),
        (
            r"Powerlaw\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)\s+improvement=([+-]?[\d.]+)%",
            "powerlaw",
        ),
        (
            r"Rayleigh\s*[—\-]\s*Best α=([\d.e+-]+)\s+RMSE=([\d.]+)\s+improvement=([+-]?[\d.]+)%",
            "rayleigh",
        ),
    ]:
        if m := re.search(pat, text, re.I):
            out["network_best"][key] = {
                "alpha": m.group(1),
                "cv_rmse": float(m.group(2)),
                "improvement_pct": float(m.group(3)),
            }
    if baseline_trials := re.findall(r"\[baseline trial\s+\d+/\d+\] state=(\w+)", text):
        out["baseline_trial_stats"] = {
            "complete": baseline_trials.count("COMPLETE"),
            "pruned": baseline_trials.count("PRUNED"),
            "total": len(baseline_trials),
        }
    if social_trials := re.findall(r"\[trial\s+\d+/\d+\] state=(\w+) rmse=", text):
        out["social_trial_stats"] = {
            "complete": social_trials.count("COMPLETE"),
            "pruned": social_trials.count("PRUNED"),
            "total": len(social_trials),
        }
    if m := re.search(r"RMSE: mean=([\d.e+-]+), best=([\d.]+)", text):
        out["search_summary"] = {
            "mean_rmse": float(m.group(1)),
            "best_rmse": float(m.group(2)),
        }
    return out


def parse_hypertune(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {"log": path.name, "completed": "[HYPERTUNE] Done." in text}
    if m := re.search(r"Best enhanced params: (\{[^}]+\})\s+RMSE=([\d.]+)", text):
        out["enhanced_cv"] = {"params": eval(m.group(1)), "rmse": float(m.group(2))}
    if m := re.search(r"Social CMF best RMSE: ([\d.]+)", text):
        out["social_cv_rmse"] = float(m.group(1))
    if m := re.search(r"Social CMF best hyperparameters:\n(\{.*?\n\})", text, re.S):
        out["social_params"] = json.loads(m.group(1))
    return out


def main() -> None:
    variants = {
        "M2": "m2_recommend.log",
        "M3": "m3_recommend.log",
        "M4a": "m4a_recommend.log",
        "M4b": "m4b_recommend.log",
        "M4c": "m4c_recommend.log",
        "M4d": "m4d_recommend.log",
        "M4c_robustness": "m4c_robustness_laplacian.log",
    }
    stage_a = {
        "M2": "m2_hypertune.log",
        "M3": "m3_hypertune.log",
        "M4a": "m4a_hypertune.log",
        "M4b": "m4b_hypertune.log",
        "M4c": "m4c_hypertune.log",
        "M4d": "m4d_hypertune.log",
    }
    results: dict = {"stage_b": {}, "stage_a": {}}
    for var, fname in variants.items():
        p = LOG_DIR / fname
        if p.exists():
            results["stage_b"][var] = parse_recommend_log(p)
    for var, fname in stage_a.items():
        p = LOG_DIR / fname
        if p.exists():
            results["stage_a"][var] = parse_hypertune(p)

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    exp = mlflow.get_experiment_by_name("mafpin")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.runName = 'recommend'",
        order_by=["start_time DESC"],
    )
    client = mlflow.MlflowClient()
    mlflow_runs = []
    for _, row in runs.head(10).iterrows():
        rid = row["run_id"]
        run = client.get_run(rid)
        p, m = run.data.params, run.data.metrics
        net = {}
        for model in ["exponential", "powerlaw", "rayleigh"]:
            hist = client.get_metric_history(rid, f"{model}_rmse_enhanced")
            vals = [h.value for h in hist if h.value is not None and 0 < h.value < 10]
            if vals:
                net[model] = {
                    "n": len(vals),
                    "mean": statistics.mean(vals),
                    "min": min(vals),
                    "median": statistics.median(vals),
                }
        mlflow_runs.append(
            {
                "start": str(row["start_time"])[:19],
                "social_mode": p.get("social_mode", "(enhanced)"),
                "social_norm": p.get("social_normalization"),
                "baseline_rmse": m.get("baseline_rmse"),
                "net": net,
            }
        )
    results["mlflow"] = mlflow_runs

    csv_summary = {}
    for model, short in [("exponential", "expo"), ("powerlaw", "power"), ("rayleigh", "ray")]:
        f = ROOT / f"data/movielens/inferred_networks/{model}/inferred_edges_{short}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="|")
        for pref in ["enhanced", "social"]:
            col = f"{pref}_rmse_mean"
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            valid = valid[(valid > 0) & (valid < 10)]
            if len(valid) == 0:
                continue
            idx = int(valid.idxmin())
            csv_summary[f"{pref}_{model}"] = {
                "n_valid": int(len(valid)),
                "min_rmse": float(valid.min()),
                "mean_rmse": float(valid.mean()),
                "best_alpha_index": idx,
                "best_alpha": float(df.loc[idx, "alpha"]) if "alpha" in df.columns else None,
            }
    results["csv"] = csv_summary
    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
