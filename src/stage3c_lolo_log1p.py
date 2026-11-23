"""Stage 3c: X098/X099 verification, leave-one-load-out robustness, and log1p HI.

1. Compare X098_DE_time inside 99.mat (normal_2.mat) with X098_DE_time inside 98.mat
   (normal_1.mat), and confirm normal_2 loads X099_DE_time (~1750 RPM at 48 kHz).
2. Leave-one-load-out (LOLO) baseline evaluation across normal_0..normal_3:
   - Baseline fit on first 70% of 3 seen loads (with 1-window gap before each holdout).
   - Threshold = mean(pooled holdout HI) + k * std(pooled holdout HI, ddof=1).
   - Unseen load treated as held-out healthy test load; faults evaluated against
     pooled healthy holdout + unseen load, and against holdout alone.
3. Replace hard z-cap with log1p compression: HI = sqrt(mean(log1p(z)^2)),
   comparing against Z_CAP and uncapped HI, and reporting large-z fractions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import de_signal_and_metadata, load_config  # noqa: E402
from stage3_hi import auroc, load_rows, split_normal_0  # noqa: E402
from stage3b_confounds import add_logs, hi_feature_names, mat  # noqa: E402

NORMAL_LABELS = ["normal", "normal_1", "normal_2", "normal_3"]
UNSEEN_ORDER = ["normal_3", "normal_2", "normal_1", "normal"]


def _shaft_peak_hz(signal: np.ndarray, rate: int, nominal_rpm: float) -> tuple[float, float]:
    """Return dominant spectral peak frequency near nominal shaft frequency (Hz)."""
    expected_hz = nominal_rpm / 60.0
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / rate)
    spec = np.abs(np.fft.rfft(signal))
    mask = (freqs >= max(expected_hz - 15.0, 1.0)) & (freqs <= expected_hz + 15.0)
    if not mask.any():
        return float("nan"), expected_hz
    peak_hz = float(freqs[mask][int(np.argmax(spec[mask]))])
    return peak_hz, expected_hz


def check_x098_and_normal_2(root: Path) -> None:
    """Item 1: verify X098_DE_time identity and normal_2 X099_DE_time shaft RPM."""
    print("=" * 100)
    print("ITEM 1: X098_DE_time (98.mat vs 99.mat) and normal_2 (X099_DE_time) check")
    print("=" * 100)

    p98 = root / "data/raw/cwru/normal_1.mat"
    p99 = root / "data/raw/cwru/normal_2.mat"
    m98 = {k: v for k, v in loadmat(p98).items() if not k.startswith("__")}
    m99 = {k: v for k, v in loadmat(p99).items() if not k.startswith("__")}

    print(f"  normal_1.mat (originally 98.mat) keys: {sorted(m98.keys())}")
    print(f"  normal_2.mat (originally 99.mat) keys: {sorted(m99.keys())}")

    if "X098_DE_time" not in m98 or "X098_DE_time" not in m99 or "X099_DE_time" not in m99:
        raise RuntimeError("Missing expected X098_DE_time or X099_DE_time key in normal_1.mat/normal_2.mat")

    a98 = np.asarray(m98["X098_DE_time"], dtype=float).squeeze()
    b98 = np.asarray(m99["X098_DE_time"], dtype=float).squeeze()
    x99 = np.asarray(m99["X099_DE_time"], dtype=float).squeeze()

    same_shape = a98.shape == b98.shape
    exact_equal = np.array_equal(a98, b98)
    max_diff = float(np.max(np.abs(a98 - b98))) if same_shape else float("nan")
    print(f"  X098_DE_time shape in 98.mat : {a98.shape}")
    print(f"  X098_DE_time shape in 99.mat : {b98.shape}")
    print(f"  X098_DE_time exact match     : {exact_equal}  (max |diff| = {max_diff:.3e})")

    if "X098_FE_time" in m98 and "X098_FE_time" in m99:
        fe98 = np.asarray(m98["X098_FE_time"], dtype=float).squeeze()
        fe99 = np.asarray(m99["X098_FE_time"], dtype=float).squeeze()
        print(f"  X098_FE_time exact match     : {np.array_equal(fe98, fe99)}  (max |diff| = {float(np.max(np.abs(fe98 - fe99))):.3e})")

    loaded_sig, _ = de_signal_and_metadata(p99)
    uses_x099 = np.array_equal(loaded_sig, x99)
    matches_x098 = np.array_equal(loaded_sig, b98)
    print(f"  preprocess.de_signal_and_metadata(normal_2.mat) loads X099_DE_time: {uses_x099}  (matches X098: {matches_x098})")
    if not uses_x099:
        raise RuntimeError("normal_2.mat did not resolve to X099_DE_time")

    peak98, exp98 = _shaft_peak_hz(b98, 48000, 1772.0)
    peak99, exp99 = _shaft_peak_hz(x99, 48000, 1750.0)
    print(f"  X098_DE_time @ 48 kHz: shaft peak = {peak98:.2f} Hz -> {peak98 * 60:.1f} RPM (nominal 1 HP = 1772 RPM / {exp98:.2f} Hz)")
    print(f"  X099_DE_time @ 48 kHz: shaft peak = {peak99:.2f} Hz -> {peak99 * 60:.1f} RPM (nominal 2 HP = 1750 RPM / {exp99:.2f} Hz)")


def _compute_scores(
    rows: list[dict], feats: list[str], mu: np.ndarray, sd: np.ndarray, z_cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (hi_log1p, hi_cap6, hi_uncap, z) for rows."""
    X = mat(rows, feats)
    z = np.maximum((X - mu) / sd, 0.0)
    hi_log1p = np.sqrt(np.mean(np.log1p(z) ** 2, axis=1))
    hi_cap6 = np.sqrt(np.mean(np.minimum(z, z_cap) ** 2, axis=1))
    hi_uncap = np.sqrt(np.mean(z ** 2, axis=1))
    for val in (hi_log1p, hi_cap6, hi_uncap, z):
        if not np.isfinite(val).all():
            raise ValueError("Non-finite values encountered while computing HI")
    return hi_log1p, hi_cap6, hi_uncap, z


def _split_pooled(
    rows: list[dict], seen_loads: list[str], fraction: float, gap: int = 1
) -> tuple[list[dict], list[dict], dict[str, tuple[int, int]]]:
    """First `fraction` + gap + holdout within each seen load, then pool."""
    base_all: list[dict] = []
    hold_all: list[dict] = []
    counts: dict[str, tuple[int, int]] = {}
    for lbl in seen_loads:
        rs = sorted([r for r in rows if r["label"] == lbl], key=lambda r: r["window_idx"])
        split = int(len(rs) * fraction)
        b = rs[:split]
        h = rs[split + gap :]
        if len(b) < 2 or len(h) < 1:
            raise ValueError(f"Insufficient windows in {lbl}: base={len(b)}, hold={len(h)}")
        base_all.extend(b)
        hold_all.extend(h)
        counts[lbl] = (len(b), len(h))
    return base_all, hold_all, counts


def evaluate_baseline_split(
    rows: list[dict],
    feats: list[str],
    base_rows: list[dict],
    hold_rows: list[dict],
    unseen_rows: list[dict] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit baseline model and compute log1p, cap6, and uncapped HI metrics."""
    z_cap = float(cfg["hi_z_cap"])
    k_thr = float(cfg["hi_threshold_k"])
    floor = float(cfg["hi_sigma_floor"])

    Xb = mat(base_rows, feats)
    mu = Xb.mean(axis=0)
    sd = np.maximum(Xb.std(axis=0, ddof=0), floor)

    hi_b_log, hi_b_cap, hi_b_unc, _ = _compute_scores(base_rows, feats, mu, sd, z_cap)
    hi_h_log, hi_h_cap, hi_h_unc, _ = _compute_scores(hold_rows, feats, mu, sd, z_cap)

    thr_log = float(hi_h_log.mean() + k_thr * hi_h_log.std(ddof=1))
    thr_cap = float(hi_h_cap.mean() + k_thr * hi_h_cap.std(ddof=1))

    unseen_metrics: dict[str, Any] = {}
    if unseen_rows:
        hi_u_log, hi_u_cap, hi_u_unc, z_u = _compute_scores(unseen_rows, feats, mu, sd, z_cap)
        q25, q75 = np.percentile(hi_u_log, [25, 75])
        unseen_metrics = dict(
            n=len(unseen_rows),
            fpr_log=float(np.mean(hi_u_log > thr_log)),
            fpr_cap=float(np.mean(hi_u_cap > thr_cap)),
            med_log=float(np.median(hi_u_log)),
            iqr_log=float(q75 - q25),
            max_log=float(hi_u_log.max()),
            med_cap=float(np.median(hi_u_cap)),
            med_unc=float(np.median(hi_u_unc)),
            frac_ge6=float(np.mean(z_u >= 6.0)),
            frac_ge20=float(np.mean(z_u >= 20.0)),
            max_z=float(z_u.max()),
            hi_log=hi_u_log,
            hi_cap=hi_u_cap,
        )
        neg_all_log = np.concatenate([hi_h_log, hi_u_log])
        neg_all_cap = np.concatenate([hi_h_cap, hi_u_cap])
    else:
        neg_all_log = hi_h_log
        neg_all_cap = hi_h_cap

    fault_labels = sorted({r["label"] for r in rows} - set(NORMAL_LABELS))
    classes: dict[str, Any] = {}
    for lbl in fault_labels:
        rs = sorted([r for r in rows if r["label"] == lbl], key=lambda r: r["window_idx"])
        hi_f_log, hi_f_cap, hi_f_unc, z_f = _compute_scores(rs, feats, mu, sd, z_cap)
        q25, q75 = np.percentile(hi_f_log, [25, 75])
        classes[lbl] = dict(
            n=len(rs),
            auroc_all=auroc(neg_all_log, hi_f_log),
            auroc_hold=auroc(hi_h_log, hi_f_log),
            recall=float(np.mean(hi_f_log > thr_log)),
            med_log=float(np.median(hi_f_log)),
            iqr_log=float(q75 - q25),
            med_cap=float(np.median(hi_f_cap)),
            med_unc=float(np.median(hi_f_unc)),
            auroc_all_cap=auroc(neg_all_cap, hi_f_cap),
            auroc_hold_cap=auroc(hi_h_cap, hi_f_cap),
            recall_cap=float(np.mean(hi_f_cap > thr_cap)),
            frac_ge6=float(np.mean(z_f >= 6.0)),
            frac_ge20=float(np.mean(z_f >= 20.0)),
            max_z=float(z_f.max()),
        )

    hold_q25, hold_q75 = np.percentile(hi_h_log, [25, 75])
    return dict(
        nb=len(base_rows),
        nh=len(hold_rows),
        thr_mean=float(hi_h_log.mean()),
        thr_std=float(hi_h_log.std(ddof=1)),
        thr=thr_log,
        thr_cap=thr_cap,
        fpr_base=float(np.mean(hi_b_log > thr_log)),
        fpr_hold=float(np.mean(hi_h_log > thr_log)),
        fpr_hold_cap=float(np.mean(hi_h_cap > thr_cap)),
        hold_med_log=float(np.median(hi_h_log)),
        hold_iqr_log=float(hold_q75 - hold_q25),
        hold_med_cap=float(np.median(hi_h_cap)),
        hold_med_unc=float(np.median(hi_h_unc)),
        unseen=unseen_metrics,
        classes=classes,
    )


def print_fold_detail(rate: int, unseen: str, seen: list[str], counts: dict[str, tuple[int, int]], res: dict[str, Any], k_thr: float) -> None:
    count_str = ", ".join(f"{l}: {counts[l][0]}/{counts[l][1]}" for l in seen)
    u = res["unseen"]
    print(f"\n{'-' * 104}")
    print(f"Rate {rate} Hz | Unseen load: {unseen} (n={u['n']}) | Seen baseline/holdout: [{count_str}]")
    print(f"{'-' * 104}")
    print(f"  Pooled baseline n={res['nb']}, pooled holdout n={res['nh']}, unseen {unseen} n={u['n']}")
    print(f"  log1p threshold = {res['thr_mean']:.4f} + {k_thr:.0f}x{res['thr_std']:.4f} = {res['thr']:.4f}  (Z_CAP=6 threshold = {res['thr_cap']:.4f})")
    print(f"  FPR pooled baseline (in-sample) : {res['fpr_base']:.3f} ({round(res['fpr_base'] * res['nb'])}/{res['nb']})")
    print(f"  FPR pooled healthy holdout      : {res['fpr_hold']:.3f} ({round(res['fpr_hold'] * res['nh'])}/{res['nh']})")
    print(
        f"  FPR unseen {unseen:<8} (alert rate): {u['fpr_log']:.3f} ({round(u['fpr_log'] * u['n'])}/{u['n']})  "
        f"[medHI={u['med_log']:.3f}, IQR={u['iqr_log']:.3f}, max={u['max_log']:.3f}]"
    )
    print(
        f"\n  {'Class':<16}{'n':>5}{'AUROC(H+U)':>12}{'AUROC(H)':>10}{'Recall':>8}"
        f"{'med(log1p)':>12}{'IQR':>8}{'med(cap6)':>11}{'med(uncap)':>12}"
    )
    for lbl, d in res["classes"].items():
        flag = "  <- weak" if (d["auroc_all"] < 0.90 or d["recall"] < 0.50) else ""
        print(
            f"  {lbl:<16}{d['n']:>5}{d['auroc_all']:>12.4f}{d['auroc_hold']:>10.4f}{d['recall']:>8.3f}"
            f"{d['med_log']:>12.3f}{d['iqr_log']:>8.3f}{d['med_cap']:>11.3f}{d['med_unc']:>12.3f}{flag}"
        )


def print_compression_table(title: str, res: dict[str, Any], include_unseen_lbl: str | None = None) -> None:
    print(f"\n  {title}")
    print(
        f"  {'Group / Class':<18}{'med(log1p)':>12}{'med(cap6)':>11}{'med(uncap)':>12}"
        f"{'frac(z>=6)':>12}{'frac(z>=20)':>13}{'max_z':>10}"
    )
    print("  " + "-" * 88)
    print(
        f"  {'holdout healthy':<18}{res['hold_med_log']:>12.3f}{res['hold_med_cap']:>11.3f}"
        f"{res['hold_med_unc']:>12.3f}{'—':>12}{'—':>13}{'—':>10}"
    )
    if include_unseen_lbl and res["unseen"]:
        u = res["unseen"]
        print(
            f"  {include_unseen_lbl + ' (unseen)':<18}{u['med_log']:>12.3f}{u['med_cap']:>11.3f}"
            f"{u['med_unc']:>12.3f}{u['frac_ge6']:>12.3f}{u['frac_ge20']:>13.3f}{u['max_z']:>10.1f}"
        )
    med_logs, med_caps, med_uncs = [], [], []
    for lbl, d in res["classes"].items():
        med_logs.append(d["med_log"])
        med_caps.append(d["med_cap"])
        med_uncs.append(d["med_unc"])
        print(
            f"  {lbl:<18}{d['med_log']:>12.3f}{d['med_cap']:>11.3f}{d['med_unc']:>12.3f}"
            f"{d['frac_ge6']:>12.3f}{d['frac_ge20']:>13.3f}{d['max_z']:>10.1f}"
        )
    print(
        f"  Fault median range: log1p [{min(med_logs):.3f}, {max(med_logs):.3f}] (span {max(med_logs)-min(med_logs):.3f}) | "
        f"cap6 [{min(med_caps):.3f}, {max(med_caps):.3f}] (span {max(med_caps)-min(med_caps):.3f}) | "
        f"uncap [{min(med_uncs):.3f}, {max(med_uncs):.3f}]"
    )


def main() -> None:
    root = Path(".")
    cfg = load_config(root / "config.yaml")
    fraction = float(cfg["baseline_fraction"])
    k_thr = float(cfg["hi_threshold_k"])
    rates = [int(r) for r in cfg["evaluation_rates"]]

    check_x098_and_normal_2(root)

    for rate in rates:
        csv_path = root / "data/processed" / f"features_{rate}.csv"
        rows, raw_cols = load_rows(csv_path)
        add_logs(rows)
        feats = hi_feature_names(raw_cols)

        for lbl in NORMAL_LABELS:
            if not any(r["label"] == lbl for r in rows):
                raise RuntimeError(f"Missing {lbl} rows in {csv_path}")

        print(f"\n\n{'=' * 104}")
        print(f"ITEM 2 & 3 @ {rate} Hz | compression={cfg.get('hi_compression')} | features ({len(feats)}): {feats}")
        print(f"{'=' * 104}")

        # Single-load normal_0 baseline reference under log1p & cap6 (compare vs wider LOLO baseline)
        b0, h0 = split_normal_0(rows, fraction, gap=1)
        ref_n0 = evaluate_baseline_split(rows, feats, b0, h0, None, cfg)
        ref_min_auc = min(d["auroc_hold"] for d in ref_n0["classes"].values())
        ref_min_rec = min(d["recall"] for d in ref_n0["classes"].values())
        print(
            f"\n  Reference single-load normal_0 baseline (log1p): thr={ref_n0['thr']:.4f}, "
            f"holdout FPR={ref_n0['fpr_hold']:.3f}, min fault AUROC={ref_min_auc:.4f}, min recall={ref_min_rec:.3f}"
        )

        # Leave-one-load-out across all 4 loads (normal_3 unseen first)
        fold_results: dict[str, dict[str, Any]] = {}
        for unseen in UNSEEN_ORDER:
            seen = [lbl for lbl in NORMAL_LABELS if lbl != unseen]
            base_rows, hold_rows, counts = _split_pooled(rows, seen, fraction, gap=1)
            unseen_rows = sorted([r for r in rows if r["label"] == unseen], key=lambda r: r["window_idx"])
            res = evaluate_baseline_split(rows, feats, base_rows, hold_rows, unseen_rows, cfg)
            fold_results[unseen] = res
            print_fold_detail(rate, unseen, seen, counts, res, k_thr)

            if unseen == "normal_3":
                # Also print 1-line comparison for old Z_CAP=6 on the normal_3-unseen fold
                cap_aucs = [d["auroc_all_cap"] for d in res["classes"].values()]
                cap_recs = [d["recall_cap"] for d in res["classes"].values()]
                print(
                    f"\n  [Comparison on unseen={unseen} with old Z_CAP=6]: thr={res['thr_cap']:.4f}, "
                    f"holdout FPR={res['fpr_hold_cap']:.3f}, unseen FPR={res['unseen']['fpr_cap']:.3f}, "
                    f"min fault AUROC(H+U)={min(cap_aucs):.4f}, min recall={min(cap_recs):.3f}"
                )

        # 4-fold LOLO summary
        print(f"\n{'=' * 104}")
        print(f"LOLO 4-FOLD SUMMARY @ {rate} Hz (log1p compression)")
        print(f"{'=' * 104}")
        print(
            f"  {'Unseen':<10}{'Base/Hold/U':>13}{'Thr':>8}{'FPR_H':>8}{'FPR_U':>8}"
            f"{'MinAUC(H+U)':>13}{'MedAUC(H+U)':>13}{'MinRec':>8}  Degraded classes vs 1.000 AUROC/recall"
        )
        print("  " + "-" * 98)
        for unseen in UNSEEN_ORDER:
            res = fold_results[unseen]
            u = res["unseen"]
            aucs = [d["auroc_all"] for d in res["classes"].values()]
            recs = [d["recall"] for d in res["classes"].values()]
            degraded = [
                f"{lbl}(AUC={d['auroc_all']:.3f},rec={d['recall']:.2f})"
                for lbl, d in res["classes"].items()
                if d["auroc_all"] < 0.999 or d["recall"] < 0.999
            ]
            deg_str = "; ".join(degraded) if degraded else "none (all >=0.999)"
            bhu = f"{res['nb']}/{res['nh']}/{u['n']}"
            print(
                f"  {unseen:<10}{bhu:>13}{res['thr']:>8.4f}{res['fpr_hold']:>8.3f}{u['fpr_log']:>8.3f}"
                f"{min(aucs):>13.4f}{float(np.median(aucs)):>13.4f}{min(recs):>8.3f}  {deg_str}"
            )

        # Item 3: log1p vs cap6 vs uncapped compression analysis
        print(f"\n{'=' * 104}")
        print(f"ITEM 3: log1p vs hard Z_CAP=6 vs uncapped @ {rate} Hz")
        print(f"{'=' * 104}")
        print_compression_table(
            "Fold A: Pooled normal_0,1,2 baseline (normal_3 unseen)",
            fold_results["normal_3"],
            include_unseen_lbl="normal_3",
        )
        print_compression_table(
            "Fold B: Single-load normal_0 baseline only (reference vs Stage 3b)",
            ref_n0,
            include_unseen_lbl=None,
        )


if __name__ == "__main__":
    main()

