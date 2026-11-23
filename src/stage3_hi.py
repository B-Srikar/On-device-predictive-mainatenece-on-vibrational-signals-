"""Stage 3: health-index construction and held-out evaluation.

HI formula  : sqrt( mean( clip(z, -Z_CAP, Z_CAP)^2 ) )
              z = (x - mu) / sigma, per feature
              mu, sigma from BASELINE only (first 70% of normal_0, sorted by window_idx)
              One window gap between baseline and holdout.
              Z_CAP = config hi_z_cap (default 6).  No covariance matrix.

Features    : rms, peak_to_peak, crest_factor, kurtosis, skewness,
              log(band_energy_* + EPS)   [log-transformed in-place]
              'std' column is dropped.

Threshold   : mean(holdout_HI) + k * std(holdout_HI, ddof=1)
              k = config hi_threshold_k (default 3). Set from healthy held-out only.

Rates       : 12000 Hz (6 bands) and 3200 Hz (4 bands), from config evaluation_rates.
normal_1-3  : used for independent FPR check if their windows appear in the CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow importing load_config from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_config  # noqa: E402

EPSILON = np.finfo(float).eps
NORMAL_0_FILE = "normal_0.mat"
NORMAL_EXTRA_FILES = {"normal_1.mat", "normal_2.mat", "normal_3.mat"}


# ── data loading ──────────────────────────────────────────────────────────────

def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None, f"{path} has no header"
        id_cols = {"file", "label", "window_idx"}
        raw_features = [c for c in reader.fieldnames if c not in id_cols]
        rows = [
            {
                "file": r["file"],
                "label": r["label"],
                "window_idx": int(r["window_idx"]),
                **{f: float(r[f]) for f in raw_features},
            }
            for r in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows, raw_features


def transform_features(
    rows: list[dict[str, Any]], raw_features: list[str]
) -> list[str]:
    """Log-transform band_energy_* columns (in-place), drop 'std'. Return new feature list."""
    band_cols = [f for f in raw_features if f.startswith("band_energy_")]
    drop_cols = {"std"}
    new_features: list[str] = []
    for f in raw_features:
        if f in drop_cols:
            continue
        if f in band_cols:
            new_features.append(f"log_{f}")
        else:
            new_features.append(f)
    for row in rows:
        for f in band_cols:
            row[f"log_{f}"] = float(np.log(max(row[f], EPSILON)))
    return new_features


def to_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    X = np.array([[r[f] for f in features] for r in rows], dtype=float)
    if not np.isfinite(X).all():
        bad = [(i, f) for i, row in enumerate(rows) for f in features if not np.isfinite(row[f])]
        raise ValueError(f"Non-finite values at {bad[:5]}")
    return X


# ── baseline split ─────────────────────────────────────────────────────────────

def split_normal_0(
    rows: list[dict[str, Any]], fraction: float = 0.70, gap: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort normal_0 by window_idx; first `fraction` → baseline, skip `gap`, rest → holdout."""
    n0 = sorted(
        [r for r in rows if r["file"] == NORMAL_0_FILE and r["label"] == "normal"],
        key=lambda r: r["window_idx"],
    )
    if not n0:
        raise ValueError(f"No normal_0 rows found (expected file='{NORMAL_0_FILE}', label='normal')")
    split = int(len(n0) * fraction)
    if split < 2:
        raise ValueError("Too few normal_0 windows for the configured baseline fraction")
    baseline = n0[:split]
    holdout = n0[split + gap :]
    if len(holdout) < 1:
        raise ValueError("No normal_0 windows left after gap; reduce fraction or gap")
    return baseline, holdout


# ── model and HI ──────────────────────────────────────────────────────────────

def fit_model(X_baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean and std from baseline (ddof=0 for population estimate)."""
    mu = X_baseline.mean(axis=0)
    sigma = X_baseline.std(axis=0, ddof=0)
    sigma = np.where(sigma < EPSILON, 1.0, sigma)  # guard zero-variance features
    return mu, sigma


def compute_hi(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z_cap: float) -> np.ndarray:
    """HI = sqrt(mean(clip(z, -z_cap, z_cap)^2))  per row."""
    z = (X - mu) / sigma
    z_clipped = np.clip(z, -z_cap, z_cap)
    return np.sqrt(np.mean(z_clipped ** 2, axis=1))


# ── AUROC ─────────────────────────────────────────────────────────────────────

def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(score(fault) > score(normal)). Ties split 0.5."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    combined = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_rate(
    config: dict[str, Any],
    rate: int,
    features_dir: Path,
    plots_dir: Path,
) -> dict[str, Any]:
    csv_path = features_dir / f"features_{rate}.csv"
    rows, raw_features = load_rows(csv_path)
    features = transform_features(rows, raw_features)
    print(f"\n{'='*70}")
    print(f"Rate: {rate} Hz    CSV: {csv_path.name}")
    print(f"Features ({len(features)}): {features}")

    fraction = float(config.get("baseline_fraction", 0.70))
    z_cap    = float(config.get("hi_z_cap", 6.0))
    k_thr    = float(config.get("hi_threshold_k", 3.0))

    # ── split ────────────────────────────────────────────────────────────────
    baseline_rows, holdout_rows = split_normal_0(rows, fraction, gap=1)
    normal_extra_rows = [r for r in rows if r["file"] in NORMAL_EXTRA_FILES]

    print(f"  normal_0 baseline  : {len(baseline_rows)} windows (first {fraction:.0%}, sorted by window_idx)")
    print(f"  gap                : 1 window")
    print(f"  normal_0 holdout   : {len(holdout_rows)} windows")
    if normal_extra_rows:
        extra_files = sorted({r["file"] for r in normal_extra_rows})
        print(f"  normal_1-3 present : {extra_files}  ({len(normal_extra_rows)} windows)")
    else:
        print(f"  normal_1-3         : not downloaded — FPR not computed")

    # ── fit model on baseline only ───────────────────────────────────────────
    X_base = to_matrix(baseline_rows, features)
    mu, sigma = fit_model(X_base)
    print(f"\n  Baseline mu   (first 4 feats): {mu[:4]}")
    print(f"  Baseline sigma (first 4 feats): {sigma[:4]}")

    # ── compute HI for every window ──────────────────────────────────────────
    X_all = to_matrix(rows, features)
    hi_all = compute_hi(X_all, mu, sigma, z_cap)
    for row, hi in zip(rows, hi_all):
        row["hi"] = float(hi)

    # ── threshold from held-out normal_0 only ────────────────────────────────
    holdout_hi = np.array([r["hi"] for r in holdout_rows])
    thr_mean  = holdout_hi.mean()
    thr_std   = holdout_hi.std(ddof=1)
    threshold = thr_mean + k_thr * thr_std
    print(f"\n  Threshold: {thr_mean:.4f} + {k_thr}×{thr_std:.4f} = {threshold:.4f}  (k={k_thr})")

    # ── FPR on held-out normal_0 ─────────────────────────────────────────────
    fpr_n0 = float(np.mean(holdout_hi > threshold))
    print(f"  FPR held-out normal_0  : {fpr_n0:.3f}  ({int(np.sum(holdout_hi > threshold))}/{len(holdout_hi)})")

    # ── FPR on normal_1-3 (if present) ───────────────────────────────────────
    fpr_extra: dict[str, float] = {}
    for fname in sorted(NORMAL_EXTRA_FILES):
        sub = [r for r in normal_extra_rows if r["file"] == fname]
        if sub:
            hi_sub = np.array([r["hi"] for r in sub])
            fpr_extra[fname] = float(np.mean(hi_sub > threshold))
    if not fpr_extra:
        print(f"  FPR normal_1-3        : n/a (files not downloaded)")

    # ── fault classes ─────────────────────────────────────────────────────────
    fault_labels = sorted({r["label"] for r in rows if r["label"] != "normal"})
    # group by label (each label = one file in this dataset)
    label_to_rows: dict[str, list[dict]] = {lbl: [] for lbl in fault_labels}
    for r in rows:
        if r["label"] in label_to_rows:
            label_to_rows[r["label"]].append(r)

    # ── per-class AUROC for HI ───────────────────────────────────────────────
    print(f"\n  Per-class HI AUROC (held-out normal_0 vs each fault class):")
    print(f"  {'Label':<22} {'Windows':>7} {'AUROC':>7} {'Median HI':>10} {'IQR':>8}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*10} {'-'*8}")
    hi_auroc: dict[str, float] = {}
    hi_median: dict[str, float] = {}
    hi_iqr: dict[str, float] = {}
    for lbl in fault_labels:
        fault_hi = np.array([r["hi"] for r in label_to_rows[lbl]])
        auc = auroc(holdout_hi, fault_hi)
        med = float(np.median(fault_hi))
        q75, q25 = float(np.percentile(fault_hi, 75)), float(np.percentile(fault_hi, 25))
        hi_auroc[lbl] = auc
        hi_median[lbl] = med
        hi_iqr[lbl] = q75 - q25
        print(f"  {lbl:<22} {len(fault_hi):>7} {auc:>7.4f} {med:>10.4f} {q75-q25:>8.4f}")
    # also print held-out normal stats
    med_n0 = float(np.median(holdout_hi))
    iqr_n0 = float(np.percentile(holdout_hi, 75) - np.percentile(holdout_hi, 25))
    print(f"  {'normal_0 holdout':<22} {len(holdout_hi):>7} {'—':>7} {med_n0:>10.4f} {iqr_n0:>8.4f}")

    # ── per-class AUROC for each single feature ───────────────────────────────
    X_holdout = to_matrix(holdout_rows, features)
    print(f"\n  Single-feature AUROC (held-out normal_0 vs each fault class):")
    # Print as label × feature table (top-6 features by mean |AUROC - 0.5| across classes)
    feat_auroc: dict[str, dict[str, float]] = {}  # feat → {label → auc}
    for fi, feat in enumerate(features):
        feat_auroc[feat] = {}
        neg_vals = X_holdout[:, fi]
        for lbl in fault_labels:
            pos_vals = to_matrix(label_to_rows[lbl], features)[:, fi]
            feat_auroc[feat][lbl] = auroc(neg_vals, pos_vals)

    # Rank features by mean |AUROC - 0.5| across all fault labels
    feat_rank = sorted(
        features,
        key=lambda f: np.mean([abs(feat_auroc[f][lbl] - 0.5) for lbl in fault_labels]),
        reverse=True,
    )
    # Print full table
    header = f"  {'Feature':<30}" + "".join(f" {lbl[:10]:>10}" for lbl in fault_labels)
    print(header)
    print("  " + "-" * (30 + 11 * len(fault_labels)))
    for feat in feat_rank:
        row_str = f"  {feat:<30}" + "".join(f" {feat_auroc[feat][lbl]:>10.4f}" for lbl in fault_labels)
        print(row_str)

    # ── plot HI per window ────────────────────────────────────────────────────
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"hi_per_window_{rate}hz.png"
    _plot_hi(rows, holdout_rows, baseline_rows, fault_labels, threshold, rate, plot_path, z_cap, k_thr)
    print(f"\n  Plot saved: {plot_path}")

    return {
        "rate": rate,
        "features": features,
        "z_cap": z_cap,
        "k_thr": k_thr,
        "baseline_n": len(baseline_rows),
        "holdout_n": len(holdout_rows),
        "threshold": threshold,
        "thr_mean": thr_mean,
        "thr_std": thr_std,
        "fpr_n0": fpr_n0,
        "fpr_extra": fpr_extra,
        "hi_auroc": hi_auroc,
        "hi_median": hi_median,
        "hi_iqr": hi_iqr,
        "feat_auroc": feat_auroc,
        "feat_rank": feat_rank,
        "fault_labels": fault_labels,
        "plot_path": plot_path,
        "normal_extra_present": bool(normal_extra_rows),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

# Colour palette: colourblind-friendly
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#999999",
    "#117733", "#882255",
]

LABEL_GROUPS = {
    "inner_race": ["inner_race_007", "inner_race_014", "inner_race_021"],
    "ball":       ["ball_007",       "ball_014",       "ball_021"],
    "outer_race": ["outer_race_007", "outer_race_014", "outer_race_021"],
}

def _plot_hi(
    all_rows: list[dict],
    holdout_rows: list[dict],
    baseline_rows: list[dict],
    fault_labels: list[str],
    threshold: float,
    rate: int,
    path: Path,
    z_cap: float,
    k_thr: float,
) -> None:
    # Build ordered list of (label, rows, color) for plotting
    series: list[tuple[str, list[dict], str]] = []
    series.append(("normal_0 baseline", baseline_rows, "#AAAAAA"))
    series.append(("normal_0 holdout",  holdout_rows,  "#000000"))
    for i, lbl in enumerate(fault_labels):
        rows_lbl = [r for r in all_rows if r["label"] == lbl]
        series.append((lbl, rows_lbl, PALETTE[i % len(PALETTE)]))

    fig, axes = plt.subplots(
        len(series), 1,
        figsize=(14, 1.6 * len(series)),
        sharex=False,
        gridspec_kw={"hspace": 0.05},
    )
    if len(series) == 1:
        axes = [axes]

    for ax, (label, rows_s, color) in zip(axes, series):
        if not rows_s:
            ax.set_visible(False)
            continue
        sorted_rows = sorted(rows_s, key=lambda r: r["window_idx"])
        x = np.arange(len(sorted_rows))
        y = np.array([r["hi"] for r in sorted_rows])
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.85)
        ax.axhline(threshold, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_ylabel(label.replace("_", "\n"), fontsize=7, rotation=0, labelpad=70, va="center")
        ax.yaxis.set_label_position("left")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(0, len(x) - 1)
        # Shade windows above threshold
        above = y > threshold
        if above.any():
            ax.fill_between(x, 0, y, where=above, color="red", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Window index (within file)", fontsize=9)
    fig.suptitle(
        f"CWRU Health Index per window @ {rate} Hz\n"
        f"HI = √(mean(clip(z,±{z_cap:.0f})²))  |  threshold = holdout_mean + {k_thr:.0f}×holdout_std  =  {threshold:.3f}",
        fontsize=10, y=1.0,
    )
    # Legend
    legend_handles = [
        mpatches.Patch(color="red", linestyle="--", label=f"Threshold = {threshold:.3f}", linewidth=0.8),
    ]
    axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",       type=Path, default=Path("config.yaml"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--plots-dir",    type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    config = load_config(args.config)
    rates = [int(r) for r in config["evaluation_rates"]]

    missing = [r for r in rates if not (args.features_dir / f"features_{r}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing CSVs for rates {missing} in {args.features_dir}. Run preprocess.py first.")

    all_results = []
    for rate in rates:
        result = evaluate_rate(config, rate, args.features_dir, args.plots_dir)
        all_results.append(result)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for res in all_results:
        rate = res["rate"]
        print(f"\n── {rate} Hz ──")
        print(f"  Z_CAP={res['z_cap']:.0f}, k={res['k_thr']:.0f}")
        print(f"  Baseline/holdout: {res['baseline_n']}/{res['holdout_n']} windows of normal_0")
        print(f"  Threshold = holdout_mean + k·holdout_std = {res['thr_mean']:.4f} + {res['k_thr']:.0f}×{res['thr_std']:.4f} = {res['threshold']:.4f}")
        print(f"  FPR held-out normal_0: {res['fpr_n0']:.3f}")
        if res["fpr_extra"]:
            for fname, fpr in res["fpr_extra"].items():
                print(f"  FPR {fname}: {fpr:.3f}")
        else:
            print(f"  FPR normal_1-3: not available (files not downloaded)")
        print(f"\n  {'Label':<22} {'HI AUROC':>9} {'Median HI':>10} {'IQR':>8}")
        print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*8}")
        for lbl in res["fault_labels"]:
            auc = res["hi_auroc"][lbl]
            med = res["hi_median"][lbl]
            iqr = res["hi_iqr"][lbl]
            flag = "  ⚠ weak" if auc < 0.70 else ""
            print(f"  {lbl:<22} {auc:>9.4f} {med:>10.4f} {iqr:>8.4f}{flag}")
        print(f"\n  Top-3 features by mean AUROC spread:")
        for feat in res["feat_rank"][:3]:
            aucs = [res["feat_auroc"][feat][lbl] for lbl in res["fault_labels"]]
            print(f"    {feat:<30}  min={min(aucs):.4f}  max={max(aucs):.4f}  mean={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()

