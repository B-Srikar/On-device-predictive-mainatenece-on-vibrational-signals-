"""Build an IMS run-to-failure health-index trend from chronological snapshots."""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from health_index import EPSILON, ema, fit_model, health_index, load_config
from preprocess import extract_features, validate_bands


def snapshot_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"IMS snapshot directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))
    if not paths:
        raise FileNotFoundError(f"No IMS snapshot files found in {directory}")
    return paths


def load_channel(path: Path, channel: int) -> np.ndarray:
    """Read an IMS whitespace-delimited snapshot and return one zero-based channel."""
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        if channel != 0:
            raise ValueError(f"{path.name} has one channel; ims_channel must be 0, not {channel}")
        signal = values
    else:
        if channel < 0 or channel >= values.shape[1]:
            raise ValueError(f"{path.name} has {values.shape[1]} channels; ims_channel={channel} is invalid")
        signal = values[:, channel]
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"{path.name} has an empty or non-finite selected channel")
    return signal


def rolling_slopes(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("ims_slope_window must be at least 2")
    slopes = np.full(len(values), np.nan)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    for end in range(window - 1, len(values)):
        y = values[end - window + 1 : end + 1]
        slopes[end] = float(np.sum(x_centered * (y - y.mean())) / denominator)
    return slopes


def first_alarm(indices: np.ndarray, scores: np.ndarray, threshold: float, start: int) -> int | None:
    eligible = (indices >= start) & np.isfinite(scores) & (scores > threshold)
    positions = np.flatnonzero(eligible)
    return None if positions.size == 0 else int(indices[positions[0]])


def write_plot(indices: np.ndarray, hi: np.ndarray, fixed_threshold: float, slope_alarm: int | None, fixed_alarm: int | None, path: Path) -> None:
    """Write a dependency-free SVG HI-vs-snapshot-index plot."""
    width, height, left, top, right, bottom = 1000, 500, 80, 50, 30, 70
    chart_width, chart_height = width - left - right, height - top - bottom
    x_min, x_max = int(indices.min()), int(indices.max())
    if x_min == x_max:
        x_max += 1
    y_min, y_max = float(min(hi.min(), fixed_threshold)), float(max(hi.max(), fixed_threshold))
    padding = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
    y_min, y_max = y_min - padding, y_max + padding

    def x_position(index: int) -> float:
        return left + (index - x_min) * chart_width / (x_max - x_min)

    def y_position(value: float) -> float:
        return top + (y_max - value) * chart_height / (y_max - y_min)

    path_data = " ".join(
        f"{'M' if position == 0 else 'L'} {x_position(int(index)):.2f} {y_position(float(value)):.2f}"
        for position, (index, value) in enumerate(zip(indices, hi))
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">IMS health index trend</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>',
        f'<path d="{path_data}" fill="none" stroke="#0072B2" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{y_position(fixed_threshold):.2f}" x2="{width - right}" y2="{y_position(fixed_threshold):.2f}" stroke="#D55E00" stroke-dasharray="5,4"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Snapshot index</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="sans-serif">EMA health index</text>',
        f'<text x="{left}" y="{height - bottom + 22}" font-family="sans-serif" font-size="12">{x_min}</text>',
        f'<text x="{width - right}" y="{height - bottom + 22}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max}</text>',
        f'<text x="{left - 8}" y="{y_position(fixed_threshold) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">fixed threshold</text>',
    ]
    for index, color, label in ((slope_alarm, "#009E73", "slope alarm"), (fixed_alarm, "#D55E00", "fixed alarm")):
        if index is not None:
            x = x_position(index)
            elements.extend((f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="{color}" stroke-dasharray="3,3"/>', f'<text x="{x + 3:.2f}" y="{top + 15}" font-family="sans-serif" font-size="12" fill="{color}">{escape(label)}: {index}</text>'))
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_outputs(rows: list[dict[str, Any]], report: dict[str, Any], processed_path: Path, report_path: Path) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with processed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    slope_alarm = "no alarm" if report["slope_alarm"] is None else str(report["slope_alarm"])
    fixed_alarm = "no alarm" if report["fixed_alarm"] is None else str(report["fixed_alarm"])
    report_path.write_text(
        "\n".join(
            [
                "# IMS run-to-failure trend",
                "",
                f"Snapshots: {report['snapshots']}; channel: {report['channel']}; baseline snapshots: first {report['baseline_snapshots']}.",
                f"Fixed-HI threshold (baseline quantile): {report['fixed_threshold']:.6g}",
                f"Slope threshold (baseline mean + {report['slope_sigma']:.6g} SD): {report['slope_threshold']:.6g}",
                f"First slope-detector alarm: {slope_alarm}",
                f"First fixed-threshold alarm: {fixed_alarm}",
                f"HI plot: `{report['plot_path']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed/ims_snapshot_features.csv"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("ims_snapshot_directory") is None or config.get("ims_channel") is None:
        raise SystemExit(
            "Error: set ims_snapshot_directory (one test folder beneath data/raw/ims/) and zero-based ims_channel in config.yaml before running."
        )
    directory = Path(str(config["ims_snapshot_directory"]))
    channel = int(config["ims_channel"])
    rate = int(config["ims_sampling_rate"])
    bands = config["fft_bands_hz"]
    validate_bands(bands, rate)
    paths = snapshot_paths(directory)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        signal = load_channel(path, channel)
        rows.append({"snapshot_index": index, "file": path.name, "sample_count": signal.size, **extract_features(signal, rate, bands)})
    feature_names = [name for name in rows[0] if name not in {"snapshot_index", "file", "sample_count"}]
    baseline_count = int(config["ims_baseline_snapshots"])
    if baseline_count < 2 or baseline_count >= len(rows):
        raise ValueError("ims_baseline_snapshots must be at least 2 and smaller than the snapshot count")
    values = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    model = fit_model(values[:baseline_count], str(config["hi_method"]), float(config["mahalanobis_regularization"]))
    hi = ema(health_index(values, model), float(config["ema_alpha"]))
    slopes = rolling_slopes(hi, int(config["ims_slope_window"]))
    baseline_slopes = slopes[:baseline_count][np.isfinite(slopes[:baseline_count])]
    if baseline_slopes.size == 0:
        raise ValueError("ims_slope_window is too large to estimate a baseline slope")
    fixed_threshold = float(np.quantile(hi[:baseline_count], float(config["hi_threshold_quantile"])))
    slope_threshold = float(baseline_slopes.mean() + float(config["ims_slope_threshold_sigma"]) * max(baseline_slopes.std(ddof=0), EPSILON))
    indices = np.arange(len(rows))
    slope_alarm = first_alarm(indices, slopes, slope_threshold, baseline_count)
    fixed_alarm = first_alarm(indices, hi, fixed_threshold, baseline_count)
    for row, hi_value, slope in zip(rows, hi, slopes):
        row["health_index_ema"] = float(hi_value)
        row["rolling_slope"] = "" if not np.isfinite(slope) else float(slope)
    plot_path = args.outputs_dir / "plots" / "ims_hi_trend.svg"
    write_plot(indices, hi, fixed_threshold, slope_alarm, fixed_alarm, plot_path)
    report = {
        "snapshots": len(rows),
        "channel": channel,
        "baseline_snapshots": baseline_count,
        "fixed_threshold": fixed_threshold,
        "slope_threshold": slope_threshold,
        "slope_sigma": float(config["ims_slope_threshold_sigma"]),
        "slope_alarm": slope_alarm,
        "fixed_alarm": fixed_alarm,
        "plot_path": plot_path,
    }
    report_path = args.outputs_dir / "ims_trend_report.md"
    write_outputs(rows, report, args.processed_path, report_path)
    print(f"First slope alarm: {slope_alarm if slope_alarm is not None else 'none'}")
    print(f"First fixed-threshold alarm: {fixed_alarm if fixed_alarm is not None else 'none'}")
    print(f"Wrote {args.processed_path}, {plot_path}, and {report_path}")


if __name__ == "__main__":
    main()
