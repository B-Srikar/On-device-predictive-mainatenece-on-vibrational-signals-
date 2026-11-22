"""Compare fixed-threshold and Isolation-Forest baselines with the HI detector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from health_index import (
    EPSILON,
    auroc,
    ema,
    fit_model,
    health_index,
    load_config,
    load_rows,
    matrix,
    split_normal,
)


def smoothed_hi(rows: list[dict[str, Any]], features: list[str], model: dict[str, Any], alpha: float) -> np.ndarray:
    """Calculate a causal EMA separately per recording, matching health_index.py."""
    scores_by_id: dict[int, float] = {}
    for filename in {row["file"] for row in rows}:
        subset = sorted((row for row in rows if row["file"] == filename), key=lambda row: row["window_idx"])
        raw = health_index(matrix(subset, features), model)
        for row, score in zip(subset, ema(raw, alpha)):
            scores_by_id[id(row)] = float(score)
    return np.array([scores_by_id[id(row)] for row in rows])


def metrics(name: str, rate: int, baseline_score: np.ndarray, normal_score: np.ndarray, fault_score: np.ndarray, threshold: float, counts: tuple[int, int, int]) -> dict[str, Any]:
    return {
        "rate_hz": rate,
        "detector": name,
        "baseline_windows": counts[0],
        "heldout_normal_windows": counts[1],
        "fault_windows": counts[2],
        "threshold": float(threshold),
        "auroc": auroc(normal_score, fault_score),
        "false_positive_rate": float(np.mean(normal_score > threshold)),
    }


def evaluate_rate(config: dict[str, Any], rate: int, features_dir: Path) -> list[dict[str, Any]]:
    rows, features = load_rows(features_dir / f"features_{rate}.csv")
    baseline_rows, holdout_rows = split_normal(rows, float(config["baseline_fraction"]))
    fault_rows = [row for row in rows if row["label"] != "normal"]
    if not fault_rows:
        raise ValueError(f"features_{rate}.csv has no fault windows")
    baseline = matrix(baseline_rows, features)
    holdout = matrix(holdout_rows, features)
    faults = matrix(fault_rows, features)
    counts = (len(baseline_rows), len(holdout_rows), len(fault_rows))

    # HI detector: identical healthy split, scaling, smoothing, and threshold rule.
    hi_model = fit_model(baseline, str(config["hi_method"]), float(config["mahalanobis_regularization"]))
    all_hi = smoothed_hi(rows, features, hi_model, float(config["ema_alpha"]))
    score_for_id = {id(row): score for row, score in zip(rows, all_hi)}
    hi_baseline = np.array([score_for_id[id(row)] for row in baseline_rows])
    hi_holdout = np.array([score_for_id[id(row)] for row in holdout_rows])
    hi_faults = np.array([score_for_id[id(row)] for row in fault_rows])
    hi_threshold = np.quantile(hi_baseline, float(config["hi_threshold_quantile"]))

    # Fixed detector: any feature beyond a fixed healthy-baseline z-score threshold.
    mean = baseline.mean(axis=0)
    std = np.where(baseline.std(axis=0, ddof=0) <= EPSILON, 1.0, baseline.std(axis=0, ddof=0))
    fixed_baseline = np.max(np.abs((baseline - mean) / std), axis=1)
    fixed_holdout = np.max(np.abs((holdout - mean) / std), axis=1)
    fixed_faults = np.max(np.abs((faults - mean) / std), axis=1)

    # Isolation Forest sees only standardized healthy baseline windows.
    forest = IsolationForest(
        n_estimators=int(config["iforest_trees"]),
        max_samples=min(int(config["iforest_subsample"]), len(baseline)),
        random_state=int(config["iforest_seed"]),
    )
    forest.fit((baseline - mean) / std)
    # score_samples is higher for inliers; negate so higher always means more anomalous.
    if_baseline = -forest.score_samples((baseline - mean) / std)
    if_holdout = -forest.score_samples((holdout - mean) / std)
    if_faults = -forest.score_samples((faults - mean) / std)
    if_threshold = np.quantile(if_baseline, float(config["iforest_threshold_quantile"]))

    return [
        metrics("health_index", rate, hi_baseline, hi_holdout, hi_faults, hi_threshold, counts),
        metrics("fixed_z_threshold", rate, fixed_baseline, fixed_holdout, fixed_faults, float(config["fixed_threshold_z"]), counts),
        metrics("isolation_forest", rate, if_baseline, if_holdout, if_faults, if_threshold, counts),
    ]


def write_outputs(results: list[dict[str, Any]], outputs_dir: Path) -> tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / "baseline_results.csv"
    columns = ["rate_hz", "detector", "baseline_windows", "heldout_normal_windows", "fault_windows", "threshold", "auroc", "false_positive_rate"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    markdown_path = outputs_dir / "baseline_results.md"
    lines = [
        "# CWRU detector comparison",
        "",
        "All detectors use the first configured fraction of the single normal recording for fitting and the remaining normal windows for testing.",
        "",
        "| Rate (Hz) | Detector | Baseline / held-out normal / fault windows | Threshold | AUROC | Held-out-normal FPR |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['rate_hz']} | {row['detector']} | {row['baseline_windows']} / {row['heldout_normal_windows']} / {row['fault_windows']} | {row['threshold']:.6g} | {row['auroc']:.4f} | {row['false_positive_rate']:.2%} |"
        for row in results
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = load_config(args.config)
    missing = [rate for rate in config["evaluation_rates"] if not (args.features_dir / f"features_{rate}.csv").is_file()]
    if missing:
        wanted = ", ".join(f"features_{rate}.csv" for rate in missing)
        raise SystemExit(f"Error: missing {wanted} in {args.features_dir}; no baseline results were written.")
    results = [row for rate in config["evaluation_rates"] for row in evaluate_rate(config, int(rate), args.features_dir)]
    csv_path, markdown_path = write_outputs(results, args.outputs_dir)
    for row in results:
        print(f"{row['rate_hz']} Hz {row['detector']}: AUROC={row['auroc']:.4f}, FPR={row['false_positive_rate']:.2%}")
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
