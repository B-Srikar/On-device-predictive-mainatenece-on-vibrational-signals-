"""Export baseline statistics for CWRU 3200 Hz (hi_baseline.h) or IMS 20480 Hz (hi_baseline_ims.h)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ims_trend import rolling_slopes  # noqa: E402
from preprocess import load_config  # noqa: E402
from stage3_hi import load_rows  # noqa: E402
from stage3b_confounds import add_logs, hi_feature_names, mat  # noqa: E402
from stage3c_lolo_log1p import _split_pooled  # noqa: E402


def c_float(value: float) -> str:
    s = f"{float(value):.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def compute_pooled_normal_0_2_baseline(config_path: Path, features_path: Path) -> dict[str, object]:
    """Compute Stage 3c pooled normal_0..normal_2 baseline & holdout threshold at 3200 Hz."""
    config = load_config(config_path)
    rows, raw_cols = load_rows(features_path)
    add_logs(rows)
    feats = hi_feature_names(raw_cols)
    if len(feats) != 8:
        raise ValueError(f"Expected 8 locked 3200 Hz HI features, found {len(feats)}: {feats}")

    seen = ["normal", "normal_1", "normal_2"]
    base_rows, hold_rows, counts = _split_pooled(rows, seen, float(config["baseline_fraction"]), gap=1)
    Xb = mat(base_rows, feats)
    Xh = mat(hold_rows, feats)

    floor = float(config["hi_sigma_floor"])
    mean = Xb.mean(axis=0)
    std = np.maximum(Xb.std(axis=0, ddof=0), floor)

    def _hi(X: np.ndarray) -> np.ndarray:
        z = np.maximum((X - mean) / std, 0.0)
        return np.sqrt(np.mean(np.log1p(z) ** 2, axis=1))

    hi_base = _hi(Xb)
    hi_hold = _hi(Xh)
    hi_thr = float(hi_hold.mean() + float(config["hi_threshold_k"]) * hi_hold.std(ddof=1))

    # Healthy-only CWRU slope threshold: rolling 12-window EMA slopes within each held-out normal load.
    alpha = float(config["ema_alpha"])
    w_slope = int(config["ims_slope_window"])
    k_slope = float(config["ims_slope_threshold_sigma"])
    holdout_slopes: list[float] = []
    for sl in seen:
        s_rows = sorted([r for r in rows if r["label"] == sl], key=lambda r: r["window_idx"])
        sp = int(len(s_rows) * float(config["baseline_fraction"]))
        h_rows = s_rows[sp + 1 :]
        if len(h_rows) < w_slope:
            continue
        hi_h_sl = _hi(mat(h_rows, feats))
        ema_h = np.empty_like(hi_h_sl)
        ema_h[0] = hi_h_sl[0]
        for t in range(1, len(hi_h_sl)):
            ema_h[t] = alpha * hi_h_sl[t] + (1.0 - alpha) * ema_h[t - 1]
        slp = rolling_slopes(ema_h, w_slope)
        holdout_slopes.extend(float(v) for v in slp[np.isfinite(slp)])
    slope_arr = np.asarray(holdout_slopes, dtype=float)
    slope_thr = float(slope_arr.mean() + k_slope * slope_arr.std(ddof=1))

    return {
        "feats": feats,
        "counts": counts,
        "n_base": len(base_rows),
        "n_hold": len(hold_rows),
        "n_hold_slopes": int(slope_arr.size),
        "mean": mean,
        "std": std,
        "median_rms": float(np.median(Xb[:, feats.index("rms")])),
        "hi_baseline_mean": float(hi_base.mean()),
        "hi_threshold": hi_thr,
        "slope_threshold": slope_thr,
    }


def compute_ims_ch1_baseline(config_path: Path, ims_csv: Path) -> dict[str, object]:
    """Compute Stage 4/5 locked Bearing 1 baseline/calibration stats (snapshots 0..981, excluding 982-983)."""
    config = load_config(config_path)
    exclude = set(int(i) for i in config["ims_exclude_snapshots"])
    rows_b1: list[dict[str, str]] = []
    with ims_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        meta = {"snapshot_index", "file", "elapsed_hours", "bearing", "channel"}
        feats = [c for c in reader.fieldnames if c not in meta]
        for row in reader:
            if row["bearing"] == "bearing_1" and int(row["snapshot_index"]) not in exclude:
                rows_b1.append(row)
    rows_b1.sort(key=lambda r: int(r["snapshot_index"]))
    X = np.array([[float(r[f]) for f in feats] for r in rows_b1], dtype=float)
    n = len(X)
    n_base = int(n * float(config["ims_baseline_fraction"]))
    n_cal = int(n * float(config["ims_calibration_fraction"]))
    cal_end = n_base + n_cal

    floor = float(config["hi_sigma_floor"])
    mean = X[:n_base].mean(axis=0)
    std = np.maximum(X[:n_base].std(axis=0, ddof=0), floor)
    z = np.maximum((X - mean) / std, 0.0)
    hi_raw = np.sqrt(np.mean(np.log1p(z) ** 2, axis=1))

    alpha = float(config["ims_ema_alpha"])
    hi_ema = np.empty_like(hi_raw)
    hi_ema[0] = hi_raw[0]
    for t in range(1, n):
        hi_ema[t] = alpha * hi_raw[t] + (1.0 - alpha) * hi_ema[t - 1]

    slopes = rolling_slopes(hi_ema, int(config["ims_slope_window"]))
    cal_slopes = slopes[n_base:cal_end]
    finite_cal = cal_slopes[np.isfinite(cal_slopes)]

    hi_thr = float(hi_ema[n_base:cal_end].mean() + float(config["ims_threshold_k"]) * hi_ema[n_base:cal_end].std(ddof=1))
    slope_thr = float(finite_cal.mean() + float(config["ims_slope_threshold_sigma"]) * finite_cal.std(ddof=1))
    return {
        "feats": feats,
        "X": X,
        "hi_raw": hi_raw,
        "hi_ema": hi_ema,
        "slopes": slopes,
        "n_snapshots": n,
        "n_base": n_base,
        "n_cal": n_cal,
        "mean": mean,
        "std": std,
        "median_rms": float(np.median(X[:n_base, feats.index("rms")])),
        "hi_baseline_mean": float(hi_ema[:n_base].mean()),
        "hi_threshold": hi_thr,
        "slope_threshold": slope_thr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cwru", "ims"), default="cwru")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "cwru":
        feat_path = args.features or Path("data/processed/features_3200.csv")
        out_path = args.output or Path("firmware/include/hi_baseline.h")
        stats = compute_pooled_normal_0_2_baseline(args.config, feat_path)
        mean: np.ndarray = stats["mean"]  # type: ignore[assignment]
        std: np.ndarray = stats["std"]  # type: ignore[assignment]
        body = "\n".join(
            (
                "#ifndef HI_BASELINE_H",
                "#define HI_BASELINE_H",
                "",
                '#include "hi_core.h"',
                "",
                f"/* Stage 3c primary fold @ 3200 Hz: pooled normal_0..2 baseline n={stats['n_base']}, holdout n={stats['n_hold']}. */",
                "static const float HI_BASELINE_MEAN[HI_FEATURE_COUNT] = {" + ", ".join(c_float(x) for x in mean) + "};",
                "static const float HI_BASELINE_STD[HI_FEATURE_COUNT] = {" + ", ".join(c_float(x) for x in std) + "};",
                f"#define HI_BASELINE_MEDIAN_RMS {c_float(stats['median_rms'])}",
                f"#define HI_BASELINE_HI_MEAN    {c_float(stats['hi_baseline_mean'])}",
                f"#define HI_THRESHOLD           {c_float(stats['hi_threshold'])}",
                f"#define HI_SLOPE_THRESHOLD     {c_float(stats['slope_threshold'])}  /* healthy normal_0-2 holdout slopes (n={stats['n_hold_slopes']}) */",
                "",
                "#endif",
                "",
            )
        )
        print(
            f"Exported CWRU pooled normal_0-2 baseline (n_base={stats['n_base']}, n_hold={stats['n_hold']}, "
            f"HI_THRESHOLD={stats['hi_threshold']:.6f}, HI_SLOPE_THRESHOLD={stats['slope_threshold']:.6f} from {stats['n_hold_slopes']} healthy holdout slopes) to {out_path}"
        )
    else:
        feat_path = args.features or Path("data/processed/ims_2nd_test_all4_features.csv")
        out_path = args.output or Path("firmware/include/hi_baseline_ims.h")
        stats = compute_ims_ch1_baseline(args.config, feat_path)
        mean = stats["mean"]  # type: ignore[assignment]
        std = stats["std"]  # type: ignore[assignment]
        body = "\n".join(
            (
                "#ifndef HI_BASELINE_IMS_H",
                "#define HI_BASELINE_IMS_H",
                "",
                '#include "hi_core.h"',
                "",
                f"/* Locked IMS Ch 1 baseline @ 20480 Hz (snapshots 0..{int(stats['n_snapshots']) - 1}, "
                f"baseline n={stats['n_base']}, cal n={stats['n_cal']}). */",
                "static const float HI_IMS_BASELINE_MEAN[HI_FEATURE_COUNT] = {" + ", ".join(c_float(x) for x in mean) + "};",
                "static const float HI_IMS_BASELINE_STD[HI_FEATURE_COUNT] = {" + ", ".join(c_float(x) for x in std) + "};",
                f"#define HI_IMS_BASELINE_MEDIAN_RMS {c_float(stats['median_rms'])}",
                f"#define HI_IMS_BASELINE_HI_MEAN    {c_float(stats['hi_baseline_mean'])}",
                f"#define HI_IMS_THRESHOLD           {c_float(stats['hi_threshold'])}",
                f"#define HI_IMS_SLOPE_THRESHOLD     {c_float(stats['slope_threshold'])}",
                "",
                "#endif",
                "",
            )
        )
        print(
            f"Exported IMS Ch 1 baseline (n={stats['n_snapshots']}, n_base={stats['n_base']}, n_cal={stats['n_cal']}, "
            f"HI_IMS_THRESHOLD={stats['hi_threshold']:.6f}, HI_IMS_SLOPE_THRESHOLD={stats['slope_threshold']:.6f}) to {out_path}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
