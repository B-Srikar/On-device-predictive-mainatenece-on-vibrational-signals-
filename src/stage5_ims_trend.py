"""Stage 5: IMS 2nd test run-to-failure health-index trend & dual-detector check.

Reads Set No. 2 chronological snapshots (20,480 Hz, 4 channels):
  Ch 1 (index 0) = Bearing 1 (outer-race failure at end of test)
  Ch 4 (index 3) = Bearing 4 (non-failing control)

Extracts window-level CWRU-style features (1024 samples, 50% overlap -> 39 windows),
log-transforms FFT band energies, excludes std and skewness, and aggregates per
snapshot using the median across windows. Fits baseline mean/std on the first
ims_baseline_fraction of snapshots, computes one-sided log1p HI + causal EMA,
calibrates fixed & rolling-slope thresholds on the next ims_calibration_fraction
snapshots only, evaluates 3-consecutive-snapshot detectors, and runs a labelled
(alpha, baseline_fraction) sensitivity sweep.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from health_index import ema  # noqa: E402
from ims_trend import rolling_slopes, snapshot_paths  # noqa: E402
from preprocess import extract_features, load_config, validate_bands, window_signal  # noqa: E402

EPSILON = np.finfo(float).eps
BASE_FEATURES = ["rms", "peak_to_peak", "crest_factor", "kurtosis"]


def parse_snapshot_times(paths: list[Path]) -> np.ndarray:
    """Parse YYYY.MM.DD.HH.MM.SS filenames and return elapsed hours from start."""
    times = [datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S") for p in paths]
    for prev, curr in zip(times, times[1:]):
        if curr <= prev:
            raise ValueError(f"Non-increasing snapshot timestamps: {prev} -> {curr}")
    t0 = times[0]
    return np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)


def extract_snapshot_features(
    paths: list[Path],
    channels: dict[str, int],
    rate: int,
    win_size: int,
    overlap: float,
    bands: list[list[float]],
    out_csv: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read each snapshot once and extract median-aggregated HI features per channel."""
    band_cols = [f"band_energy_{int(lo)}_{int(hi)}hz" for lo, hi in bands]
    feat_names = [*BASE_FEATURES, *[f"log_{c}" for c in band_cols]]
    matrices = {name: np.empty((len(paths), len(feat_names)), dtype=float) for name in channels}
    csv_rows: list[dict[str, Any]] = []

    t0 = datetime.strptime(paths[0].name, "%Y.%m.%d.%H.%M.%S")
    for idx, path in enumerate(paths):
        data = np.loadtxt(path, dtype=float)
        if data.ndim != 2 or data.shape[0] != rate or data.shape[1] < max(channels.values()) + 1:
            raise ValueError(f"{path.name}: unexpected shape {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError(f"{path.name}: contains NaN or Inf")
        elapsed_h = (datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S") - t0).total_seconds() / 3600.0

        for label, ch in channels.items():
            sig = data[:, ch]
            win_feats = []
            for w in window_signal(sig, win_size, overlap):
                raw_f = extract_features(w, rate, bands)
                vec = [raw_f[f] for f in BASE_FEATURES] + [
                    float(np.log(max(raw_f[c], EPSILON))) for c in band_cols
                ]
                win_feats.append(vec)
            med_vec = np.median(np.asarray(win_feats, dtype=float), axis=0)
            if not np.isfinite(med_vec).all():
                raise ValueError(f"{path.name} ({label}): non-finite median features")
            matrices[label][idx] = med_vec
            csv_rows.append(
                {
                    "snapshot_index": idx,
                    "file": path.name,
                    "elapsed_hours": round(elapsed_h, 6),
                    "bearing": label,
                    "channel": ch,
                    **{name: float(val) for name, val in zip(feat_names, med_vec)},
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["snapshot_index", "file", "elapsed_hours", "bearing", "channel", *feat_names]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return matrices, feat_names


def consecutive_runs(cond: np.ndarray, m: int) -> np.ndarray:
    """Return boolean array where index t is True iff cond[t-m+1 : t+1] are all True."""
    trig = np.zeros(len(cond), dtype=bool)
    for t in range(m - 1, len(cond)):
        trig[t] = bool(cond[t - m + 1 : t + 1].all())
    return trig


def evaluate_bearing(
    X: np.ndarray,
    t_hours: np.ndarray,
    cfg: dict[str, Any],
    alpha: float,
    baseline_frac: float,
) -> dict[str, Any]:
    """Compute baseline stats, log1p HI + EMA, calibration thresholds, and detector triggers."""
    n = len(X)
    cal_frac = float(cfg["ims_calibration_fraction"])
    floor = float(cfg["hi_sigma_floor"])
    k_hi = float(cfg["ims_threshold_k"])
    w_slope = int(cfg["ims_slope_window"])
    k_slope = float(cfg["ims_slope_threshold_sigma"])
    m_consec = int(cfg["ims_consecutive_alarms"])

    n_base = int(n * baseline_frac)
    n_cal = int(n * cal_frac)
    cal_end = n_base + n_cal
    if n_base < 2 or n_cal < 2 or cal_end >= n:
        raise ValueError(f"Invalid block sizes: n={n}, n_base={n_base}, n_cal={n_cal}")

    mu = X[:n_base].mean(axis=0)
    sigma = np.maximum(X[:n_base].std(axis=0, ddof=0), floor)
    z = np.maximum((X - mu) / sigma, 0.0)
    hi_raw = np.sqrt(np.mean(np.log1p(z) ** 2, axis=1))
    hi_ema = ema(hi_raw, alpha)
    slopes = rolling_slopes(hi_ema, w_slope)

    if not np.isfinite(hi_raw).all() or not np.isfinite(hi_ema).all():
        raise ValueError("Non-finite HI values encountered")

    hi_cal = hi_ema[n_base:cal_end]
    hi_thr = float(hi_cal.mean() + k_hi * hi_cal.std(ddof=1))

    cal_slopes = slopes[n_base:cal_end]
    finite_cal_slopes = cal_slopes[np.isfinite(cal_slopes)]
    if finite_cal_slopes.size < 2 or not np.isfinite(finite_cal_slopes).all():
        raise ValueError("Insufficient finite slopes in calibration block")
    slope_thr = float(finite_cal_slopes.mean() + k_slope * finite_cal_slopes.std(ddof=1))
    hi_base_mean = float(hi_ema[:n_base].mean())

    cond_fixed = hi_ema > hi_thr
    cond_slope = np.isfinite(slopes) & (slopes > slope_thr) & (hi_ema > hi_base_mean)
    trig_fixed = consecutive_runs(cond_fixed, m_consec)
    trig_slope = consecutive_runs(cond_slope, m_consec)

    def _first_post(trig: np.ndarray) -> int | None:
        pos = np.flatnonzero(trig[cal_end:])
        return None if pos.size == 0 else int(cal_end + pos[0])

    first_fixed = _first_post(trig_fixed)
    first_slope = _first_post(trig_slope)

    def _lead_h(idx: int | None) -> float | None:
        return None if idx is None else float(t_hours[-1] - t_hours[idx])

    rho_ema, p_ema = spearmanr(t_hours, hi_ema)
    rho_raw, p_raw = spearmanr(t_hours, hi_raw)
    return {
        "n": n,
        "n_base": n_base,
        "n_cal": n_cal,
        "cal_end": cal_end,
        "alpha": alpha,
        "baseline_frac": baseline_frac,
        "hi_raw": hi_raw,
        "hi_ema": hi_ema,
        "slopes": slopes,
        "hi_base_mean": hi_base_mean,
        "hi_cal_mean": float(hi_cal.mean()),
        "hi_cal_std": float(hi_cal.std(ddof=1)),
        "hi_thr": hi_thr,
        "slope_cal_mean": float(finite_cal_slopes.mean()),
        "slope_cal_std": float(finite_cal_slopes.std(ddof=1)),
        "slope_thr": slope_thr,
        "trig_fixed": trig_fixed,
        "trig_slope": trig_slope,
        "base_alarms_fixed": int(trig_fixed[:n_base].sum()),
        "base_alarms_slope": int(trig_slope[:n_base].sum()),
        "cal_alarms_fixed": int(trig_fixed[n_base:cal_end].sum()),
        "cal_alarms_slope": int(trig_slope[n_base:cal_end].sum()),
        "healthy_alarms_fixed": int(trig_fixed[:cal_end].sum()),
        "healthy_alarms_slope": int(trig_slope[:cal_end].sum()),
        "post_alarms_fixed": int(trig_fixed[cal_end:].sum()),
        "post_alarms_slope": int(trig_slope[cal_end:].sum()),
        "first_fixed": first_fixed,
        "first_slope": first_slope,
        "lead_fixed_h": _lead_h(first_fixed),
        "lead_slope_h": _lead_h(first_slope),
        "rho_ema": float(rho_ema),
        "p_ema": float(p_ema),
        "rho_raw": float(rho_raw),
        "p_raw": float(p_raw),
        "end_hi_ema": float(hi_ema[-1]),
        "max_hi_ema": float(hi_ema.max()),
    }


def _fmt_alarm(idx: int | None, lead_h: float | None, paths: list[Path], t_hours: np.ndarray) -> str:
    if idx is None or lead_h is None:
        return "no alarm"
    return f"idx {idx} ({paths[idx].name}, t={t_hours[idx]:.2f} h, lead={lead_h:.2f} h)"


def write_trend_plot(
    results: dict[str, dict[str, Any]], t_hours: np.ndarray, out_path: Path
) -> None:
    """Plot HI vs snapshot index for Bearing 1 (failed) and Bearing 4 (control)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    titles = {
        "bearing_1": "Bearing 1 (Channel 1 / idx 0) — Outer Race Failure at End of Test",
        "bearing_4": "Bearing 4 (Channel 4 / idx 3) — Non-Failing Control",
    }
    for ax, key in zip(axes, ("bearing_1", "bearing_4")):
        res = results[key]
        x = np.arange(res["n"])
        n_base, cal_end = res["n_base"], res["cal_end"]
        ax.axvspan(0, n_base - 1, color="#D5E8D4", alpha=0.5, label=f"Baseline fit (0..{n_base - 1})")
        ax.axvspan(n_base, cal_end - 1, color="#FFF2CC", alpha=0.6, label=f"Calibration ({n_base}..{cal_end - 1})")
        ax.plot(x, res["hi_raw"], color="#B0B0B0", lw=0.7, alpha=0.75, label="Raw log1p HI")
        ax.plot(x, res["hi_ema"], color="#0072B2", lw=1.4, label=f"EMA HI (alpha={res['alpha']})")
        ax.axhline(res["hi_thr"], color="#D55E00", ls="--", lw=1.1, label=f"Fixed threshold ({res['hi_thr']:.3f})")
        if res["first_fixed"] is not None:
            ax.axvline(
                res["first_fixed"],
                color="#D55E00",
                ls=":",
                lw=1.3,
                label=f"First fixed alarm: idx {res['first_fixed']} (lead {res['lead_fixed_h']:.1f} h)",
            )
        if res["first_slope"] is not None:
            ax.axvline(
                res["first_slope"],
                color="#009E73",
                ls="-.",
                lw=1.3,
                label=f"First slope alarm: idx {res['first_slope']} (lead {res['lead_slope_h']:.1f} h)",
            )
        ax.set_title(titles[key], fontsize=10, loc="left")
        ax.set_ylabel("log1p HI", fontsize=9)
        ax.set_xlim(0, res["n"] - 1)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2)
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[-1].set_xlabel(f"Chronological snapshot index (0..{res['n'] - 1}, ~10 min/step, total {t_hours[-1]:.2f} h)", fontsize=9)
    fig.suptitle("IMS 2nd Test Run-to-Failure Trend (20,480 Hz native, median snapshot features)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(".")
    cfg = load_config(root / "config.yaml")

    snapshot_dir = root / str(cfg["ims_snapshot_directory"])
    rate = int(cfg["ims_sampling_rate"])
    win_size = int(cfg["ims_window_size"])
    overlap = float(cfg["ims_overlap"])
    bands = cfg["ims_fft_bands_hz"]
    validate_bands(bands, rate)

    channels = {
        "bearing_1": int(cfg["ims_channel"]),
        "bearing_4": int(cfg["ims_control_channel"]),
    }

    paths = snapshot_paths(snapshot_dir)
    t_hours = parse_snapshot_times(paths)
    win_count = sum(1 for _ in window_signal(np.zeros(rate), win_size, overlap))

    csv_path = root / "data/processed/ims_2nd_test_features.csv"
    matrices, feat_names = extract_snapshot_features(paths, channels, rate, win_size, overlap, bands, csv_path)

    alpha = float(cfg["ims_ema_alpha"])
    base_frac = float(cfg["ims_baseline_fraction"])

    print("=" * 100)
    print("STAGE 5: IMS 2ND TEST RUN-TO-FAILURE TREND")
    print("=" * 100)
    print("Readme verification (Set No. 2): 984 ASCII snapshots, 4 channels (Bearing 1=Ch 1 [idx 0] failed")
    print("with outer-race defect; Bearings 2-4 [Ch 2-4, idx 1-3] did not fail).")
    print(f"Chronological span: {paths[0].name} -> {paths[-1].name} ({len(paths)} snapshots, {t_hours[-1]:.2f} h)")
    print(f"Signal processing : native {rate} Hz, window={win_size}, overlap={overlap} ({win_count} windows/snapshot -> median)")
    print(f"Bands (Hz)        : {bands}")
    print(f"Features ({len(feat_names)})      : {feat_names}")
    print(
        f"Locked config     : baseline={base_frac:.0%}, calibration={float(cfg['ims_calibration_fraction']):.0%}, "
        f"EMA alpha={alpha}, k_hi={cfg['ims_threshold_k']}, slope_win={cfg['ims_slope_window']}, "
        f"k_slope={cfg['ims_slope_threshold_sigma']}, consec={cfg['ims_consecutive_alarms']}"
    )
    print(f"Saved snapshot features to: {csv_path}")

    results: dict[str, dict[str, Any]] = {}
    for key, label in (("bearing_1", "Bearing 1 (Ch 1 / idx 0 — FAILED)"), ("bearing_4", "Bearing 4 (Ch 4 / idx 3 — CONTROL)")):
        res = evaluate_bearing(matrices[key], t_hours, cfg, alpha=alpha, baseline_frac=base_frac)
        results[key] = res
        print(f"\n{'-' * 100}")
        print(label)
        print(f"{'-' * 100}")
        print(
            f"  Blocks            : baseline 0..{res['n_base'] - 1} (n={res['n_base']}), "
            f"calibration {res['n_base']}..{res['cal_end'] - 1} (n={res['n_cal']}), "
            f"monitor {res['cal_end']}..{res['n'] - 1} (n={res['n'] - res['cal_end']})"
        )
        print(
            f"  HI threshold      : {res['hi_cal_mean']:.4f} + {cfg['ims_threshold_k']}x{res['hi_cal_std']:.4f} "
            f"= {res['hi_thr']:.4f}   (baseline mean HI={res['hi_base_mean']:.4f})"
        )
        print(
            f"  Slope threshold   : {res['slope_cal_mean']:.6f} + {cfg['ims_slope_threshold_sigma']}x{res['slope_cal_std']:.6f} "
            f"= {res['slope_thr']:.6f}"
        )
        print(
            f"  Alarms (3 consec) : healthy 0..{res['cal_end'] - 1} -> fixed={res['healthy_alarms_fixed']} "
            f"(base={res['base_alarms_fixed']}, cal={res['cal_alarms_fixed']}), "
            f"slope={res['healthy_alarms_slope']} (base={res['base_alarms_slope']}, cal={res['cal_alarms_slope']})"
        )
        print(
            f"                      post-calibration {res['cal_end']}..{res['n'] - 1} -> "
            f"fixed={res['post_alarms_fixed']}, slope={res['post_alarms_slope']}"
        )
        print(f"  Detector (a) fixed: {_fmt_alarm(res['first_fixed'], res['lead_fixed_h'], paths, t_hours)}")
        print(f"  Detector (b) slope: {_fmt_alarm(res['first_slope'], res['lead_slope_h'], paths, t_hours)}")
        print(
            f"  Spearman rho      : HI_ema vs time = {res['rho_ema']:.4f} (p={res['p_ema']:.3e}) | "
            f"HI_raw vs time = {res['rho_raw']:.4f} (p={res['p_raw']:.3e})"
        )
        print(f"  HI summary (EMA)  : max={res['max_hi_ema']:.4f}, final={res['end_hi_ema']:.4f}")

    plot_path = root / "outputs/plots/ims_2nd_test_hi_trend.png"
    write_trend_plot(results, t_hours, plot_path)
    print(f"\nSaved HI trend plot to: {plot_path}")

    # Sensitivity table (labelled separately; primary config.yaml untouched)
    alphas = [float(a) for a in cfg["ims_sensitivity_alphas"]]
    base_fracs = [float(b) for b in cfg["ims_sensitivity_baseline_fractions"]]
    print(f"\n{'=' * 100}")
    print("SENSITIVITY TABLE (alpha in [0.05, 0.1, 0.2] x baseline_fraction in [5%, 10%, 15%]; cal=10% fixed)")
    print(f"{'=' * 100}")
    print(
        f"  {'Alpha':>5} {'Base%':>6} | {'B1 Hlth(F/S)':>12} {'B1 Fixed(idx/lead)':>20} {'B1 Slope(idx/lead)':>20} {'B1 rho':>7} | "
        f"{'B4 Hlth(F/S)':>12} {'B4 Post(F/S)':>12} {'B4 rho':>7}"
    )
    print("  " + "-" * 112)
    for b_frac in base_fracs:
        for a in alphas:
            r1 = evaluate_bearing(matrices["bearing_1"], t_hours, cfg, alpha=a, baseline_frac=b_frac)
            r4 = evaluate_bearing(matrices["bearing_4"], t_hours, cfg, alpha=a, baseline_frac=b_frac)
            b1_f = f"{r1['first_fixed']}/{r1['lead_fixed_h']:.1f}h" if r1["first_fixed"] is not None else "none"
            b1_s = f"{r1['first_slope']}/{r1['lead_slope_h']:.1f}h" if r1["first_slope"] is not None else "none"
            tag = " *" if (abs(a - alpha) < 1e-9 and abs(b_frac - base_frac) < 1e-9) else ""
            print(
                f"  {a:>5.2f} {b_frac * 100:>5.0f}% | "
                f"{r1['healthy_alarms_fixed']:>5}/{r1['healthy_alarms_slope']:<6} "
                f"{b1_f:>20} {b1_s:>20} {r1['rho_ema']:>7.4f} | "
                f"{r4['healthy_alarms_fixed']:>5}/{r4['healthy_alarms_slope']:<6} "
                f"{r4['post_alarms_fixed']:>5}/{r4['post_alarms_slope']:<6} {r4['rho_ema']:>7.4f}{tag}"
            )
    print("  (* marks the locked primary config.yaml setting)")
    print("\nNote: Single run-to-failure recording (IMS Set No. 2); results do not generalise across bearings/machines.")


if __name__ == "__main__":
    main()

