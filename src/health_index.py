"""Fit and evaluate an ESP32-portable vibration health index from feature CSVs."""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from preprocess import load_config


IDENTIFIER_COLUMNS = {"file", "label", "window_idx"}
EPSILON = np.finfo(float).eps


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        features = [field for field in reader.fieldnames if field not in IDENTIFIER_COLUMNS]
        if not features:
            raise ValueError(f"{path} has no feature columns")
        rows = []
        for row in reader:
            rows.append(
                {
                    "file": row["file"],
                    "label": row["label"],
                    "window_idx": int(row["window_idx"]),
                    **{feature: float(row[feature]) for feature in features},
                }
            )
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows, features


def split_normal(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normal = sorted((row for row in rows if row["label"] == "normal"), key=lambda row: row["window_idx"])
    normal_files = {row["file"] for row in normal}
    if not normal:
        raise ValueError("No healthy rows with label 'normal'")
    if len(normal_files) != 1:
        raise ValueError(f"Expected one normal recording, found {sorted(normal_files)}")
    split = int(len(normal) * fraction)
    if split < 2 or len(normal) - split < 1:
        raise ValueError("Normal recording is too short for the configured baseline/hold-out split")
    return normal[:split], normal[split:]


def matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    values = np.array([[row[name] for name in features] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature CSV contains NaN or infinity")
    return values


def fit_model(baseline: np.ndarray, method: str, regularization: float) -> dict[str, np.ndarray | str]:
    mean = baseline.mean(axis=0)
    std = baseline.std(axis=0, ddof=0)
    safe_std = np.where(std <= EPSILON, 1.0, std)
    model: dict[str, np.ndarray | str] = {"mean": mean, "std": safe_std, "method": method}
    if method == "mahalanobis":
        covariance = np.cov(baseline, rowvar=False, ddof=0)
        covariance = np.atleast_2d(covariance) + regularization * np.eye(baseline.shape[1])
        model["inverse_covariance"] = np.linalg.pinv(covariance)
    elif method != "zscore_mean":
        raise ValueError("hi_method must be 'zscore_mean' or 'mahalanobis'")
    return model


def health_index(values: np.ndarray, model: dict[str, np.ndarray | str]) -> np.ndarray:
    centered = values - np.asarray(model["mean"])
    if model["method"] == "mahalanobis":
        inverse = np.asarray(model["inverse_covariance"])
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0.0))
    z_scores = centered / np.asarray(model["std"])
    return z_scores.mean(axis=1)


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def auroc(negative: np.ndarray, positive: np.ndarray) -> float:
    """AUROC for fault-positive scores, including average ranks for ties."""
    values = np.concatenate((negative, positive))
    labels = np.concatenate((np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (len(positive) * len(negative)))


def feature_separation(baseline: np.ndarray, faults: np.ndarray, features: list[str]) -> list[tuple[str, float, float]]:
    results = []
    for index, name in enumerate(features):
        auc = auroc(baseline[:, index], faults[:, index])
        effect = (faults[:, index].mean() - baseline[:, index].mean()) / max(baseline[:, index].std(ddof=0), EPSILON)
        results.append((name, auc, float(effect)))
    return sorted(results, key=lambda item: abs(item[1] - 0.5), reverse=True)


def plot_hi(rows: list[dict[str, Any]], holdout: list[dict[str, Any]], rate: int, plot_path: Path) -> None:
    """Write a dependency-free SVG line chart of held-out normal and fault HIs."""
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    series = []
    for filename in sorted({row["file"] for row in rows}):
        subset = [row for row in rows if row["file"] == filename]
        if subset[0]["label"] == "normal":
            subset, label = holdout, f"{filename} (held-out normal)"
        else:
            label = filename
        series.append((label, subset))
    x_values = [row["window_idx"] for _, subset in series for row in subset]
    y_values = [row["hi_ema"] for _, subset in series for row in subset]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    else:
        padding = 0.05 * (y_max - y_min)
        y_min -= padding
        y_max += padding
    width, height, left, top, right, bottom = 1000, 500, 80, 50, 30, 70
    chart_width, chart_height = width - left - right, height - top - bottom

    def point(row: dict[str, Any]) -> tuple[float, float]:
        x = left + (row["window_idx"] - x_min) * chart_width / (x_max - x_min)
        y = top + (y_max - row["hi_ema"]) * chart_height / (y_max - y_min)
        return x, y

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">CWRU health index at {rate} Hz</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Window index</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="sans-serif">EMA health index</text>',
        f'<text x="{left}" y="{height - bottom + 22}" font-family="sans-serif" font-size="12">{x_min}</text>',
        f'<text x="{width - right}" y="{height - bottom + 22}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max}</text>',
        f'<text x="{left - 8}" y="{height - bottom}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.3g}</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.3g}</text>',
    ]
    for index, (label, subset) in enumerate(series):
        color = palette[index % len(palette)]
        path = " ".join(f"{'M' if row_index == 0 else 'L'} {x:.2f} {y:.2f}" for row_index, row in enumerate(subset) for x, y in [point(row)])
        legend_y = top + 20 * index
        elements.extend(
            [
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>',
                f'<line x1="{width - 240}" y1="{legend_y}" x2="{width - 220}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{width - 215}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(label)}</text>',
            ]
        )
    elements.append("</svg>")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(elements), encoding="utf-8")


def evaluate_rate(config: dict[str, Any], rate: int, input_dir: Path, plots_dir: Path) -> dict[str, Any]:
    feature_path = input_dir / f"features_{rate}.csv"
    rows, features = load_rows(feature_path)
    baseline_rows, holdout_rows = split_normal(rows, float(config["baseline_fraction"]))
    baseline = matrix(baseline_rows, features)
    holdout = matrix(holdout_rows, features)
    faults_rows = [row for row in rows if row["label"] != "normal"]
    if not faults_rows:
        raise ValueError(f"{feature_path}: no fault rows")
    faults = matrix(faults_rows, features)
    model = fit_model(baseline, str(config["hi_method"]), float(config["mahalanobis_regularization"]))
    for row, value in zip(rows, health_index(matrix(rows, features), model)):
        row["hi_ema"] = float(value)
    for filename in {row["file"] for row in rows}:
        subset = sorted((row for row in rows if row["file"] == filename), key=lambda row: row["window_idx"])
        smoothed = ema(np.array([row["hi_ema"] for row in subset]), float(config["ema_alpha"]))
        for row, value in zip(subset, smoothed):
            row["hi_ema"] = float(value)
    baseline_ema = np.array([row["hi_ema"] for row in baseline_rows])
    holdout_ema = np.array([row["hi_ema"] for row in holdout_rows])
    fault_ema = np.array([row["hi_ema"] for row in faults_rows])
    threshold = float(np.quantile(baseline_ema, float(config["hi_threshold_quantile"])))
    per_fault_auc = {}
    for filename in sorted({row["file"] for row in faults_rows}):
        file_rows = [row for row in rows if row["file"] == filename]
        file_values = np.array([row["hi_ema"] for row in file_rows])
        per_fault_auc[filename] = auroc(np.array([row["hi_ema"] for row in holdout_rows]), file_values)
    plot_path = plots_dir / f"health_index_{rate}.svg"
    plot_hi(rows, holdout_rows, rate, plot_path)
    return {
        "rate": rate,
        "baseline_windows": len(baseline_rows),
        "holdout_windows": len(holdout_rows),
        "fault_windows": len(faults_rows),
        "threshold": threshold,
        "fpr": float(np.mean(holdout_ema > threshold)),
        "overall_auc": auroc(holdout_ema, fault_ema),
        "per_fault_auc": per_fault_auc,
        "features": feature_separation(baseline, faults, features),
        "plot_path": plot_path,
    }


def write_report(results: list[dict[str, Any]], path: Path, config: dict[str, Any]) -> None:
    lines = [
        "# CWRU health-index evaluation",
        "",
        f"Method: `{config['hi_method']}`; EMA alpha: `{config['ema_alpha']}`; baseline: first `{config['baseline_fraction']:.0%}` of normal windows.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['rate']} Hz",
                "",
                f"Baseline / held-out normal / fault windows: {result['baseline_windows']} / {result['holdout_windows']} / {result['fault_windows']}",
                f"HI threshold (baseline {config['hi_threshold_quantile']:.0%} quantile): {result['threshold']:.6g}",
                f"Held-out-normal false-positive rate: {result['fpr']:.2%}",
                f"Overall AUROC (normal hold-out vs all faults): {result['overall_auc']:.4f}",
                "",
                "| Fault file | AUROC |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {name} | {value:.4f} |" for name, value in result["per_fault_auc"].items())
        lines.extend(["", "Top separating features (univariate fault-vs-baseline):", "", "| Feature | AUROC | Effect (baseline SD) |", "| --- | ---: | ---: |"])
        lines.extend(f"| {name} | {auc:.4f} | {effect:.4g} |" for name, auc, effect in result["features"])
        lines.extend(["", f"Plot: `{result['plot_path']}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
        raise SystemExit(
            f"Error: missing {wanted} in {args.features_dir}. Run preprocess.py once with --rate-mode original "
            "and once with --rate-mode configured after the MAT files are available."
        )
    results = [evaluate_rate(config, int(rate), args.features_dir, args.outputs_dir / "plots") for rate in config["evaluation_rates"]]
    report_path = args.outputs_dir / "health_index_report.md"
    write_report(results, report_path, config)
    for result in results:
        print(
            f"{result['rate']} Hz: AUROC={result['overall_auc']:.4f}, "
            f"held-out-normal FPR={result['fpr']:.2%}, plot={result['plot_path']}"
        )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
