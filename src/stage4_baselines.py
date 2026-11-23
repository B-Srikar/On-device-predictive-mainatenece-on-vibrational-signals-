"""Stage 4: baseline comparisons (B1-B3) & HI ablations on CWRU and IMS 2nd test.

Pre-specified Stage 4 rules (from config.yaml; zero tuning on faults/failure):
  - CWRU: 4 leave-one-load-out folds at 12 kHz and 3200 Hz. Compare B1_rms,
    B1_kurtosis, B2_any_feature, B3_iforest (100 trees, max_samples=256,
    random_state=0), and log1p_HI. Thresholds = mean + 3 std of healthy holdout;
    also compute recall at 1% and 5% FPR on strictly healthy windows.
  - IMS: exclude snapshots 982-983 (shaft stopped); machine-off gate skips update
    if snapshot RMS < 10% of baseline median RMS; presumed-healthy false-alarm
    window = snapshots 196-491; 3-consecutive rule for B1-B3 + HI_raw, HI_ema,
    HI_ema_slope ablations across all 4 channels (Bearing 1 primary).
  - Exploratory (clearly labelled): full kurtosis z trajectory for Bearings 1-4.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent))
from ims_trend import rolling_slopes  # noqa: E402
from preprocess import load_config  # noqa: E402
from stage3_hi import auroc, load_rows  # noqa: E402
from stage3b_confounds import add_logs, hi_feature_names, mat  # noqa: E402
from stage3c_lolo_log1p import NORMAL_LABELS, UNSEEN_ORDER, _split_pooled  # noqa: E402
from stage5_ims_trend import consecutive_runs  # noqa: E402

CWRU_DETECTORS = ["B1_rms", "B1_kurtosis", "B2_any_feature", "B3_iforest", "log1p_HI"]
IMS_DETECTORS = [
    "B1_rms",
    "B1_kurtosis",
    "B2_any_feature",
    "B3_iforest",
    "HI_raw",
    "HI_ema",
    "HI_ema_slope",
]
BEARING_KEYS = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
BEARING_LABELS = {
    "bearing_1": "Bearing 1 (Ch 1 — Failed OR)",
    "bearing_2": "Bearing 2 (Ch 2 — Normal)",
    "bearing_3": "Bearing 3 (Ch 3 — Normal)",
    "bearing_4": "Bearing 4 (Ch 4 — Normal)",
}
BEARING_COLORS = {
    "bearing_1": "#D55E00",
    "bearing_2": "#0072B2",
    "bearing_3": "#009E73",
    "bearing_4": "#CC79A7",
}


# ── shared baseline detector fitting & scoring ─────────────────────────────────

def fit_base_detectors(
    X_base: np.ndarray, feats: list[str], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Fit mu/sd scaling and IsolationForest on healthy baseline windows only."""
    floor = float(cfg["hi_sigma_floor"])
    mu = X_base.mean(axis=0)
    sd = np.maximum(X_base.std(axis=0, ddof=0), floor)
    Z_base = (X_base - mu) / sd

    forest = IsolationForest(
        n_estimators=int(cfg["stage4_iforest_n_estimators"]),
        max_samples=min(int(cfg["stage4_iforest_max_samples"]), len(X_base)),
        random_state=int(cfg["stage4_iforest_random_state"]),
    )
    forest.fit(Z_base)
    total_nodes = int(sum(est.tree_.node_count for est in forest.estimators_))

    return {
        "feats": feats,
        "mu": mu,
        "sd": sd,
        "rms_idx": feats.index("rms"),
        "kurt_idx": feats.index("kurtosis"),
        "forest": forest,
        "iforest_total_nodes": total_nodes,
        "iforest_max_samples": min(int(cfg["stage4_iforest_max_samples"]), len(X_base)),
        "iforest_trees": len(forest.estimators_),
    }


def score_base_detectors(X: np.ndarray, model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return raw score arrays for B1_rms, B1_kurtosis, B2_any_feature, B3_iforest, log1p_HI."""
    Z = (X - model["mu"]) / model["sd"]
    Z_pos = np.maximum(Z, 0.0)
    scores = {
        "B1_rms": Z[:, model["rms_idx"]],
        "B1_kurtosis": Z[:, model["kurt_idx"]],
        "B2_any_feature": np.max(Z_pos, axis=1),
        "B3_iforest": -model["forest"].decision_function(Z),
        "log1p_HI": np.sqrt(np.mean(np.log1p(Z_pos) ** 2, axis=1)),
    }
    for name, arr in scores.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in detector '{name}'")
    return scores


def param_count(detector: str, n_feats: int, model: dict[str, Any]) -> int:
    """Scalar float parameters stored on-device (or total tree nodes for Isolation Forest)."""
    if detector in ("B1_rms", "B1_kurtosis"):
        return 3  # mu_1, sd_1, thr
    if detector in ("B2_any_feature", "log1p_HI"):
        return 2 * n_feats + 1  # mu[F], sd[F], thr
    if detector == "B3_iforest":
        return int(model["iforest_total_nodes"])
    raise ValueError(detector)


# ── Part A: CWRU ───────────────────────────────────────────────────────────────

def run_cwru(cfg: dict[str, Any], out_csv: Path) -> list[dict[str, Any]]:
    fraction = float(cfg["baseline_fraction"])
    k_thr = float(cfg["hi_threshold_k"])
    rates = [int(r) for r in cfg["evaluation_rates"]]
    all_records: list[dict[str, Any]] = []

    print("=" * 116)
    print("PART A: CWRU LEAVE-ONE-LOAD-OUT BASELINES (B1-B3 vs log1p_HI) AT 12 kHz & 3200 Hz")
    print("=" * 116)
    print(
        f"Thresholds = mean + {k_thr:g} std (healthy holdout); 1%/5% FPR thresholds from healthy (holdout + unseen). "
        f"IForest = ({cfg['stage4_iforest_n_estimators']} trees, max_samples={cfg['stage4_iforest_max_samples']}, "
        f"random_state={cfg['stage4_iforest_random_state']})."
    )

    for rate in rates:
        csv_path = Path("data/processed") / f"features_{rate}.csv"
        rows, raw_cols = load_rows(csv_path)
        add_logs(rows)
        feats = hi_feature_names(raw_cols)
        fault_labels = sorted({r["label"] for r in rows} - set(NORMAL_LABELS))

        rate_records: list[dict[str, Any]] = []
        fold_stats: list[dict[str, Any]] = []

        for unseen in UNSEEN_ORDER:
            seen = [lbl for lbl in NORMAL_LABELS if lbl != unseen]
            base_rows, hold_rows, _ = _split_pooled(rows, seen, fraction, gap=1)
            unseen_rows = sorted([r for r in rows if r["label"] == unseen], key=lambda r: r["window_idx"])

            X_base = mat(base_rows, feats)
            X_hold = mat(hold_rows, feats)
            X_unseen = mat(unseen_rows, feats)

            model = fit_base_detectors(X_base, feats, cfg)
            s_hold_all = score_base_detectors(X_hold, model)
            s_unseen_all = score_base_detectors(X_unseen, model)

            fault_scores_by_lbl = {
                lbl: score_base_detectors(
                    mat(sorted([r for r in rows if r["label"] == lbl], key=lambda r: r["window_idx"]), feats),
                    model,
                )
                for lbl in fault_labels
            }

            for det in CWRU_DETECTORS:
                sh = s_hold_all[det]
                su = s_unseen_all[det]
                s_healthy = np.concatenate([sh, su])
                thr = float(sh.mean() + k_thr * sh.std(ddof=1))
                thr_1pct = float(np.quantile(s_healthy, 0.99))
                thr_5pct = float(np.quantile(s_healthy, 0.95))
                fpr_hold = float(np.mean(sh > thr))
                fpr_unseen = float(np.mean(su > thr))

                aucs, rec_t, rec_1, rec_5 = [], [], [], []
                for lbl in fault_labels:
                    sf = fault_scores_by_lbl[lbl][det]
                    auc = auroc(s_healthy, sf)
                    rt = float(np.mean(sf > thr))
                    r1 = float(np.mean(sf > thr_1pct))
                    r5 = float(np.mean(sf > thr_5pct))
                    aucs.append(auc)
                    rec_t.append(rt)
                    rec_1.append(r1)
                    rec_5.append(r5)
                    rec = {
                        "rate_hz": rate,
                        "n_features": len(feats),
                        "unseen_load": unseen,
                        "detector": det,
                        "fault_class": lbl,
                        "n_base": len(base_rows),
                        "n_hold": len(hold_rows),
                        "n_unseen": len(unseen_rows),
                        "n_fault": len(sf),
                        "threshold": thr,
                        "thr_1pct_fpr": thr_1pct,
                        "thr_5pct_fpr": thr_5pct,
                        "fpr_hold": fpr_hold,
                        "fpr_unseen": fpr_unseen,
                        "auroc": auc,
                        "recall_at_thr": rt,
                        "recall_at_1pct_fpr": r1,
                        "recall_at_5pct_fpr": r5,
                        "model_size": param_count(det, len(feats), model),
                    }
                    rate_records.append(rec)
                    all_records.append(rec)

                fold_stats.append(
                    {
                        "unseen": unseen,
                        "detector": det,
                        "n_bhu": f"{len(base_rows)}/{len(hold_rows)}/{len(unseen_rows)}",
                        "thr": thr,
                        "fpr_hold": fpr_hold,
                        "fpr_unseen": fpr_unseen,
                        "min_auc": min(aucs),
                        "mean_auc": float(np.mean(aucs)),
                        "min_rec_thr": min(rec_t),
                        "mean_rec_thr": float(np.mean(rec_t)),
                        "mean_rec_1pct": float(np.mean(rec_1)),
                        "mean_rec_5pct": float(np.mean(rec_5)),
                        "size": param_count(det, len(feats), model),
                        "if_max_samples": model["iforest_max_samples"],
                    }
                )

        # Print Model Size & 4-Fold Summary for this rate
        print(f"\n{'-' * 116}")
        print(f"CWRU @ {rate} Hz (F={len(feats)} features)")
        print(f"{'-' * 116}")
        print("  Model sizes across the 4 LOLO folds:")
        for det in CWRU_DETECTORS:
            sizes = [fs["size"] for fs in fold_stats if fs["detector"] == det]
            if det == "B3_iforest":
                ms = fold_stats[0]["if_max_samples"]
                print(
                    f"    {det:<16}: {sizes[0]}..{max(sizes)} total tree nodes "
                    f"(mean {np.mean(sizes):.1f} across 100 trees, max_samples={ms}; + {2 * len(feats) + 1} scale/thr floats)"
                )
            else:
                print(f"    {det:<16}: {sizes[0]} scalar floats (mu, sd, thr)")

        print("\n  4-Fold Mean across LOLO folds (each metric averaged over 4 folds x 9 classes; min across all 36 pairs shown too):")
        print(
            f"    {'Detector':<16}{'FPR_H(mean)':>12}{'FPR_U(mean/max)':>17}"
            f"{'AUROC(mean/min)':>17}{'Rec@thr(mean/min)':>19}{'Rec@1%FPR(mean)':>17}{'Rec@5%FPR(mean)':>17}"
        )
        print("    " + "-" * 115)
        for det in CWRU_DETECTORS:
            fdet = [fs for fs in fold_stats if fs["detector"] == det]
            rdet = [r for r in rate_records if r["detector"] == det]
            print(
                f"    {det:<16}{np.mean([f['fpr_hold'] for f in fdet]):>12.3f}"
                f"{np.mean([f['fpr_unseen'] for f in fdet]):>9.3f}/{max(f['fpr_unseen'] for f in fdet):.3f}  "
                f"{np.mean([r['auroc'] for r in rdet]):>8.4f}/{min(r['auroc'] for r in rdet):.4f}  "
                f"{np.mean([r['recall_at_thr'] for r in rdet]):>9.3f}/{min(r['recall_at_thr'] for r in rdet):.3f}  "
                f"{np.mean([r['recall_at_1pct_fpr'] for r in rdet]):>15.3f}  "
                f"{np.mean([r['recall_at_5pct_fpr'] for r in rdet]):>15.3f}"
            )

        print("\n  Per-fold detail (FPR on holdout / unseen load, min & mean AUROC, min recall@thr):")
        print(f"    {'Unseen':<10}{'B/H/U':>13}  {'Detector':<16}{'Thr':>8}{'FPR_H':>8}{'FPR_U':>8}{'MinAUC':>9}{'MeanAUC':>9}{'MinRec':>8}{'MeanRec':>9}{'Rec@1%':>8}{'Rec@5%':>8}")
        for fs in fold_stats:
            print(
                f"    {fs['unseen']:<10}{fs['n_bhu']:>13}  {fs['detector']:<16}{fs['thr']:>8.3f}{fs['fpr_hold']:>8.3f}"
                f"{fs['fpr_unseen']:>8.3f}{fs['min_auc']:>9.4f}{fs['mean_auc']:>9.4f}{fs['min_rec_thr']:>8.3f}"
                f"{fs['mean_rec_thr']:>9.3f}{fs['mean_rec_1pct']:>8.3f}{fs['mean_rec_5pct']:>8.3f}"
            )

        print(f"\n  Per-class 4-fold mean @ {rate} Hz [AUROC | Rec@thr | Rec@1%FPR | Rec@5%FPR]:")
        print(f"    {'Class':<16}" + "".join(f"{d:>26}" for d in CWRU_DETECTORS))
        print("    " + "-" * (16 + 26 * len(CWRU_DETECTORS)))
        for lbl in fault_labels:
            cells = []
            for det in CWRU_DETECTORS:
                sub = [r for r in rate_records if r["fault_class"] == lbl and r["detector"] == det]
                cell = (
                    f"{np.mean([r['auroc'] for r in sub]):.3f}/"
                    f"{np.mean([r['recall_at_thr'] for r in sub]):.2f}/"
                    f"{np.mean([r['recall_at_1pct_fpr'] for r in sub]):.2f}/"
                    f"{np.mean([r['recall_at_5pct_fpr'] for r in sub]):.2f}"
                )
                cells.append(f"{cell:>26}")
            print(f"    {lbl:<16}" + "".join(cells))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n  Saved CWRU Stage 4 results CSV: {out_csv} ({len(all_records)} rows)")
    return all_records


# ── Part B: IMS ────────────────────────────────────────────────────────────────

def load_ims_all4(csv_path: Path) -> tuple[dict[str, np.ndarray], list[str], np.ndarray, list[str]]:
    """Load ims_2nd_test_all4_features.csv grouped by bearing."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        meta = {"snapshot_index", "file", "elapsed_hours", "bearing", "channel"}
        feats = [c for c in reader.fieldnames if c not in meta]
        grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in BEARING_KEYS}
        for row in reader:
            grouped[row["bearing"]].append(row)
    matrices: dict[str, np.ndarray] = {}
    t_hours: np.ndarray | None = None
    files: list[str] | None = None
    for key in BEARING_KEYS:
        rows_b = sorted(grouped[key], key=lambda r: int(r["snapshot_index"]))
        matrices[key] = np.array([[float(r[f]) for f in feats] for r in rows_b], dtype=float)
        if t_hours is None:
            t_hours = np.array([float(r["elapsed_hours"]) for r in rows_b], dtype=float)
            files = [r["file"] for r in rows_b]
    assert t_hours is not None and files is not None
    return matrices, feats, t_hours, files


def run_ims(cfg: dict[str, Any], out_csv: Path, kurt_plot_path: Path) -> list[dict[str, Any]]:
    csv_in = Path("data/processed/ims_2nd_test_all4_features.csv")
    if not csv_in.is_file():
        raise FileNotFoundError(f"Missing {csv_in}; run Stage 5b first.")

    matrices_full, feats, t_full, files_full = load_ims_all4(csv_in)
    exclude = set(int(i) for i in cfg["ims_exclude_snapshots"])
    keep_mask = np.array([i not in exclude for i in range(len(t_full))], dtype=bool)
    off_ratio = float(cfg["ims_machine_off_rms_ratio"])
    hlth_lo, hlth_hi = [int(v) for v in cfg["ims_presumed_healthy_range"]]

    base_frac = float(cfg["ims_baseline_fraction"])
    cal_frac = float(cfg["ims_calibration_fraction"])
    alpha = float(cfg["ims_ema_alpha"])
    k_hi = float(cfg["ims_threshold_k"])
    w_slope = int(cfg["ims_slope_window"])
    k_slope = float(cfg["ims_slope_threshold_sigma"])
    m_consec = int(cfg["ims_consecutive_alarms"])

    t_hours = t_full[keep_mask]
    files = [f for f, k in zip(files_full, keep_mask) if k]
    n = len(t_hours)
    n_base = int(n * base_frac)
    n_cal = int(n * cal_frac)
    cal_end = n_base + n_cal
    rms_idx = feats.index("rms")
    kurt_idx = feats.index("kurtosis")

    print(f"\n\n{'=' * 116}")
    print("PART B: IMS 2ND TEST BASELINES & HI ABLATIONS (all 4 channels; snapshots 982-983 excluded)")
    print("=" * 116)
    print(
        f"Snapshots kept: {n} (0..{n - 1}, span {files[0]} -> {files[-1]}, {t_hours[-1]:.2f} h); excluded={sorted(exclude)} | "
        f"baseline=0..{n_base - 1}, cal={n_base}..{cal_end - 1}, presumed_healthy={hlth_lo}..{hlth_hi} (n={hlth_hi - hlth_lo + 1}), "
        f"monitor={cal_end}..{n - 1} | consec={m_consec}"
    )

    records: list[dict[str, Any]] = []
    kurt_z_all: dict[str, np.ndarray] = {}

    for key in BEARING_KEYS:
        X_full = matrices_full[key]
        rms_off_thr = off_ratio * float(np.median(X_full[:n_base, rms_idx]))
        full_gated_idx = [int(i) for i in np.flatnonzero(X_full[:, rms_idx] < rms_off_thr)]

        X = X_full[keep_mask]
        gated = X[:, rms_idx] < rms_off_thr
        if gated[0]:
            raise RuntimeError(f"{key}: snapshot 0 triggers machine-off gate")

        model = fit_base_detectors(X[:n_base], feats, cfg)
        base_scores = score_base_detectors(X, model)
        kurt_z_all[key] = base_scores["B1_kurtosis"]

        # Gated causal EMA on log1p HI (initialized with first raw value)
        hi_raw = base_scores["log1p_HI"]
        hi_ema = np.empty_like(hi_raw)
        hi_ema[0] = hi_raw[0]
        for t in range(1, n):
            if gated[t]:
                hi_ema[t] = hi_ema[t - 1]
            else:
                hi_ema[t] = alpha * hi_raw[t] + (1.0 - alpha) * hi_ema[t - 1]

        slopes = rolling_slopes(hi_ema, w_slope)
        cal_slopes = slopes[n_base:cal_end]
        finite_cal = cal_slopes[np.isfinite(cal_slopes)]
        slope_thr = float(finite_cal.mean() + k_slope * finite_cal.std(ddof=1))
        hi_ema_base_mean = float(hi_ema[:n_base].mean())

        det_series: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for det in ("B1_rms", "B1_kurtosis", "B2_any_feature", "B3_iforest", "HI_raw"):
            arr = hi_raw if det == "HI_raw" else base_scores[det]
            thr = float(arr[n_base:cal_end].mean() + k_hi * arr[n_base:cal_end].std(ddof=1))
            det_series[det] = (arr, (arr > thr) & (~gated), thr)

        thr_ema = float(hi_ema[n_base:cal_end].mean() + k_hi * hi_ema[n_base:cal_end].std(ddof=1))
        det_series["HI_ema"] = (hi_ema, (hi_ema > thr_ema) & (~gated), thr_ema)
        det_series["HI_ema_slope"] = (
            slopes,
            np.isfinite(slopes) & (slopes > slope_thr) & (hi_ema > hi_ema_base_mean) & (~gated),
            slope_thr,
        )

        print(f"\n{'-' * 116}")
        print(
            f"{BEARING_LABELS[key]} | rms_off_thr={rms_off_thr:.5f} | gated in 0..981: {int(gated.sum())} "
            f"(in full 0..983: {full_gated_idx}) | IForest nodes={model['iforest_total_nodes']}"
        )
        print(f"{'-' * 116}")
        print(
            f"  {'Detector':<16}{'Threshold':>11}{'Cal(98-195)':>13}{'Hlth(196-491)':>15}"
            f"{'Post(196-981)':>15}{'FirstIdx(>=196)':>17}{'Timestamp':>22}{'Lead(h)':>10}"
        )
        print("  " + "-" * 114)
        for det in IMS_DETECTORS:
            _, cond, thr = det_series[det]
            trig = consecutive_runs(cond, m_consec)
            cal_alarms = int(trig[n_base:cal_end].sum())
            hlth_alarms = int(trig[hlth_lo : hlth_hi + 1].sum())
            post_alarms = int(trig[cal_end:].sum())
            pos = np.flatnonzero(trig[cal_end:])
            first_idx = None if pos.size == 0 else int(cal_end + pos[0])
            lead_h = None if first_idx is None else float(t_hours[-1] - t_hours[first_idx])
            ts = "none" if first_idx is None else files[first_idx]
            idx_str = "none" if first_idx is None else str(first_idx)
            lead_str = "none" if lead_h is None else f"{lead_h:.2f}"
            print(
                f"  {det:<16}{thr:>11.5f}{cal_alarms:>13}{hlth_alarms:>15}"
                f"{post_alarms:>15}{idx_str:>17}{ts:>22}{lead_str:>10}"
            )
            records.append(
                {
                    "bearing": key,
                    "detector": det,
                    "n_snapshots": n,
                    "gated_in_0_981": int(gated.sum()),
                    "threshold": thr,
                    "cal_alarms_98_195": cal_alarms,
                    "healthy_alarms_196_491": hlth_alarms,
                    "post_cal_alarms_196_981": post_alarms,
                    "first_alarm_idx": "" if first_idx is None else first_idx,
                    "first_alarm_file": "" if first_idx is None else files[first_idx],
                    "lead_time_h": "" if lead_h is None else round(lead_h, 4),
                    "model_size": model["iforest_total_nodes"]
                    if det == "B3_iforest"
                    else (3 if det in ("B1_rms", "B1_kurtosis") else 2 * len(feats) + 1),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  Saved IMS Stage 4 results CSV: {out_csv} ({len(records)} rows)")

    # ── Exploratory Kurtosis Z Trajectory ─────────────────────────────────────
    print(f"\n{'=' * 116}")
    print("EXPLORATORY (POST-HOC DIAGNOSTIC — NOT A PRIMARY METHOD): FULL KURTOSIS Z TRAJECTORY (0..981)")
    print("=" * 116)
    print(
        f"  {'Bearing':<28}{'Hlth med/max(196-491)':>23}{'Late med/p95/max(525-981)':>29}"
        f"{'z@450':>8}{'z@600':>8}{'z@800':>8}{'z@950':>8}{'End(981)':>10}"
    )
    print("  " + "-" * 120)
    for key in BEARING_KEYS:
        kz = kurt_z_all[key]
        h = kz[hlth_lo : hlth_hi + 1]
        l = kz[525:]
        print(
            f"  {BEARING_LABELS[key]:<28}"
            f"{np.median(h):>9.2f} / {np.max(h):<10.2f}  "
            f"{np.median(l):>7.2f} / {np.percentile(l, 95):>6.2f} / {np.max(l):<8.2f}  "
            f"{kz[450]:>8.2f}{kz[600]:>8.2f}{kz[800]:>8.2f}{kz[950]:>8.2f}{kz[-1]:>10.2f}"
        )

    # Plot exploratory kurtosis z trajectory for all 4 bearings
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.5), sharex=False)
    for ax, zoom_start, subtitle in (
        (axes[0], 0, f"Full test excluding 982-983 (snapshots 0..{n - 1}, {t_hours[-1]:.2f} h)"),
        (axes[1], 400, f"Onset & degradation zoom (snapshots 400..{n - 1})"),
    ):
        if zoom_start < cal_end:
            ax.axvspan(0, n_base - 1, color="#D5E8D4", alpha=0.45, label=f"Baseline (0..{n_base - 1})")
            ax.axvspan(n_base, cal_end - 1, color="#FFF2CC", alpha=0.55, label=f"Calibration ({n_base}..{cal_end - 1})")
        if zoom_start <= hlth_hi:
            ax.axvspan(max(hlth_lo, zoom_start), hlth_hi, color="#DAE8FC", alpha=0.35, label=f"Presumed healthy ({hlth_lo}..{hlth_hi})")
        for key in BEARING_KEYS:
            ax.plot(
                x[zoom_start:],
                kurt_z_all[key][zoom_start:],
                color=BEARING_COLORS[key],
                lw=1.1 if key == "bearing_1" else 0.85,
                alpha=0.9 if key == "bearing_1" else 0.75,
                label=BEARING_LABELS[key],
            )
        ax.axhline(3.0, color="black", ls="--", lw=0.9, alpha=0.7, label="z = +3.0 reference")
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.set_ylabel("Kurtosis z-score", fontsize=9)
        ax.set_xlim(zoom_start, n - 1)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2)

    axes[1].set_xlabel(f"Chronological snapshot index (0..{n - 1}, 982-983 excluded)", fontsize=9)
    fig.suptitle("EXPLORATORY: IMS 2nd Test Full Kurtosis z Trajectory for Bearings 1-4", fontsize=11)
    fig.tight_layout()
    kurt_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(kurt_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exploratory kurtosis z plot: {kurt_plot_path}")
    return records


def main() -> None:
    cfg = load_config(Path("config.yaml"))
    run_cwru(cfg, Path("outputs/stage4_cwru_results.csv"))
    run_ims(
        cfg,
        Path("outputs/stage4_ims_results.csv"),
        Path("outputs/plots/ims_stage4_exploratory_kurtosis_z.png"),
    )


if __name__ == "__main__":
    main()

