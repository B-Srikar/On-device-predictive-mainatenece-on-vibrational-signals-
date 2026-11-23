"""Stage 5b: IMS 2nd test 4-channel diagnostics, decomposition, and raw inspection.

Does not modify config.yaml or data/raw/. Uses locked Stage 5 parameters:
  baseline=10%, calibration=10%, EMA alpha=0.1 (initialized with first raw value),
  k_hi=3.0, slope_win=12, k_slope=3.0, consec=3.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from ims_trend import snapshot_paths  # noqa: E402
from preprocess import load_config, validate_bands  # noqa: E402
from stage5_ims_trend import (  # noqa: E402
    evaluate_bearing,
    extract_snapshot_features,
    parse_snapshot_times,
)

EPSILON = np.finfo(float).eps
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


def compute_decomposition(
    X: np.ndarray, res: dict[str, Any], feat_names: list[str], cfg: dict[str, Any], snap_indices: list[int]
) -> dict[int, dict[str, Any]]:
    """Compute per-feature z, log1p(z)^2, and % HI^2 share at specified snapshots."""
    n_base = res["n_base"]
    floor = float(cfg["hi_sigma_floor"])
    mu = X[:n_base].mean(axis=0)
    sigma = np.maximum(X[:n_base].std(axis=0, ddof=0), floor)
    out: dict[int, dict[str, Any]] = {}
    for idx in snap_indices:
        z = np.maximum((X[idx] - mu) / sigma, 0.0)
        contrib = np.log1p(z) ** 2
        total_c = float(contrib.sum())
        shares = 100.0 * contrib / total_c if total_c > 0 else np.zeros_like(contrib)
        out[idx] = {
            "z": z,
            "contrib": contrib,
            "shares": shares,
            "hi_raw": float(res["hi_raw"][idx]),
            "hi_ema": float(res["hi_ema"][idx]),
        }
    return out


def raw_snapshot_stats(paths: list[Path], indices: list[int], rate: int) -> dict[int, list[dict[str, float]]]:
    """Compute raw time-domain stats per channel for selected snapshot indices."""
    out: dict[int, list[dict[str, float]]] = {}
    for idx in indices:
        data = np.loadtxt(paths[idx], dtype=float)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(f"{paths[idx].name}: unexpected raw shape {data.shape}")
        ch_stats = []
        for ch in range(4):
            sig = data[:, ch]
            cen = sig - sig.mean()
            sd = float(np.std(cen))
            kurt = 0.0 if sd <= EPSILON else float(kurtosis(cen, fisher=False, bias=False))
            ch_stats.append(
                {
                    "len": float(sig.size),
                    "min": float(sig.min()),
                    "max": float(sig.max()),
                    "mean": float(sig.mean()),
                    "std": float(sig.std()),
                    "rms": float(np.sqrt(np.mean(sig ** 2))),
                    "kurt": kurt,
                }
            )
        out[idx] = ch_stats
    return out


def plot_all4(
    results: dict[str, dict[str, Any]],
    t_hours: np.ndarray,
    out_path: Path,
    title_suffix: str,
) -> None:
    """Two-panel plot of Bearings 1-4 EMA HI: full range (top) and 400..end zoom (bottom)."""
    n = results["bearing_1"]["n"]
    n_base = results["bearing_1"]["n_base"]
    cal_end = results["bearing_1"]["cal_end"]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=False)
    for ax, zoom_start, subtitle in (
        (axes[0], 0, f"Full test (snapshots 0..{n - 1}, {t_hours[-1]:.2f} h)"),
        (axes[1], 400, f"Onset & degradation zoom (snapshots 400..{n - 1})"),
    ):
        if zoom_start < cal_end:
            ax.axvspan(0, n_base - 1, color="#D5E8D4", alpha=0.45, label=f"Baseline (0..{n_base - 1})")
            ax.axvspan(n_base, cal_end - 1, color="#FFF2CC", alpha=0.55, label=f"Calibration ({n_base}..{cal_end - 1})")
        for key in BEARING_KEYS:
            res = results[key]
            color = BEARING_COLORS[key]
            fa = res["first_fixed"]
            fa_str = f"fix@{fa}" if fa is not None else "no fix"
            ax.plot(
                x[zoom_start:],
                res["hi_ema"][zoom_start:],
                color=color,
                lw=1.3 if key == "bearing_1" else 1.0,
                alpha=0.95 if key == "bearing_1" else 0.80,
                label=f"{BEARING_LABELS[key]} ({fa_str})",
            )
            if fa is not None and fa >= zoom_start:
                ax.axvline(fa, color=color, ls=":", lw=1.0, alpha=0.7)
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.set_ylabel("EMA log1p HI", fontsize=9)
        ax.set_xlim(zoom_start, n - 1)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2)

    axes[1].set_xlabel(f"Chronological snapshot index ({t_hours[-1]:.2f} h total)", fontsize=9)
    fig.suptitle(f"IMS 2nd Test: All 4 Bearings EMA HI {title_suffix}", fontsize=11)
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
    alpha = float(cfg["ims_ema_alpha"])
    base_frac = float(cfg["ims_baseline_fraction"])

    paths = snapshot_paths(snapshot_dir)
    t_hours = parse_snapshot_times(paths)
    channels = {key: i for i, key in enumerate(BEARING_KEYS)}
    csv_all4 = root / "data/processed/ims_2nd_test_all4_features.csv"
    matrices, feat_names = extract_snapshot_features(
        paths, channels, rate, win_size, overlap, bands, csv_all4
    )

    # Evaluate all 4 channels (full 0..983)
    results = {
        key: evaluate_bearing(matrices[key], t_hours, cfg, alpha=alpha, baseline_frac=base_frac)
        for key in BEARING_KEYS
    }
    for key, res in results.items():
        if not np.isclose(res["hi_ema"][0], res["hi_raw"][0], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{key}: EMA[0]={res['hi_ema'][0]} != raw[0]={res['hi_raw'][0]}")

    # ── ITEM 1 ────────────────────────────────────────────────────────────────
    print("=" * 108)
    print("ITEM 1: ALL 4 BEARINGS THROUGH LOCKED STAGE 5 PIPELINE (snapshots 0..983)")
    print("=" * 108)
    print(f"Saved all-4 snapshot median features to: {csv_all4}")
    print(
        f"{'Bearing':<28}{'HI_thr':>8}{'Slp_thr':>10}{'Hlth(F/S)':>11}{'Post(F/S)':>11}"
        f"{'FirstFixed(idx/lead)':>22}{'FirstSlope(idx/lead)':>22}{'MaxHI':>8}{'EndHI':>8}"
    )
    print("-" * 126)
    for key in BEARING_KEYS:
        r = results[key]
        hlth = f"{r['healthy_alarms_fixed']}/{r['healthy_alarms_slope']}"
        post = f"{r['post_alarms_fixed']}/{r['post_alarms_slope']}"
        ff = f"{r['first_fixed']} / {r['lead_fixed_h']:.2f}h" if r["first_fixed"] is not None else "none"
        fs = f"{r['first_slope']} / {r['lead_slope_h']:.2f}h" if r["first_slope"] is not None else "none"
        print(
            f"{BEARING_LABELS[key]:<28}{r['hi_thr']:>8.4f}{r['slope_thr']:>10.6f}{hlth:>11}{post:>11}"
            f"{ff:>22}{fs:>22}{r['max_hi_ema']:>8.3f}{r['end_hi_ema']:>8.3f}"
        )

    # ── ITEM 2 ────────────────────────────────────────────────────────────────
    snap_list = [450, 600, 800, 950]
    print(f"\n{'=' * 108}")
    print("ITEM 2: FEATURE DECOMPOSITION AT SNAPSHOTS 450, 600, 800, 950 (Bearing 1 vs Bearing 4)")
    print("=" * 108)
    for key in ("bearing_1", "bearing_4"):
        dec = compute_decomposition(matrices[key], results[key], feat_names, cfg, snap_list)
        print(f"\n--- {BEARING_LABELS[key]} ---")
        for idx in snap_list:
            d = dec[idx]
            time_share = float(d["shares"][:4].sum())
            band_shares = d["shares"][4:]
            print(
                f"  Snapshot {idx} ({paths[idx].name}, t={t_hours[idx]:.2f}h): "
                f"HI_raw={d['hi_raw']:.4f}, HI_ema={d['hi_ema']:.4f} | "
                f"time-domain share={time_share:.1f}%, FFT bands share={100.0 - time_share:.1f}%"
            )
            print(f"    {'Feature':<32}{'z_one_sided':>12}{'log1p(z)^2':>12}{'Share(%)':>10}")
            for fname, zv, cv, sv in zip(feat_names, d["z"], d["contrib"], d["shares"]):
                print(f"    {fname:<32}{zv:>12.3f}{cv:>12.4f}{sv:>10.1f}")
            print(
                "    Bands breakdown: "
                + ", ".join(f"{fn.replace('log_band_energy_', '')}={sv:.1f}%" for fn, sv in zip(feat_names[4:], band_shares))
            )

    # ── ITEM 3 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 108}")
    print("ITEM 3: CORRELATION BETWEEN BEARING 1 AND BEARING 4 TRAJECTORIES")
    print("=" * 108)
    rms1 = matrices["bearing_1"][:, feat_names.index("rms")]
    rms4 = matrices["bearing_4"][:, feat_names.index("rms")]
    hi1_raw, hi4_raw = results["bearing_1"]["hi_raw"], results["bearing_4"]["hi_raw"]
    hi1_ema, hi4_ema = results["bearing_1"]["hi_ema"], results["bearing_4"]["hi_ema"]
    print(f"  {'Window slice':<18}{'Series':<12}{'n':>5}{'Pearson r':>12}{'p_pearson':>13}{'Spearman rho':>14}{'p_spearman':>13}")
    print("  " + "-" * 85)
    for label_sl, sl in (("196..983", slice(196, 984)), ("525..983", slice(525, 984))):
        for sname, s1, s4 in (("rms", rms1, rms4), ("HI_raw", hi1_raw, hi4_raw), ("HI_ema", hi1_ema, hi4_ema)):
            pr, pp = pearsonr(s1[sl], s4[sl])
            sr, sp = spearmanr(s1[sl], s4[sl])
            print(f"  {label_sl:<18}{sname:<12}{len(s1[sl]):>5}{pr:>12.4f}{pp:>13.3e}{sr:>14.4f}{sp:>13.3e}")

    # ── ITEM 4 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 108}")
    print("ITEM 4: RAW SNAPSHOT INSPECTION — LAST 5 (979..983) & FIRST 60 (0..59)")
    print("=" * 108)
    last5_idx = list(range(len(paths) - 5, len(paths)))
    steady_idx = list(range(200, 451))
    late_idx = list(range(950, 979))
    first60_idx = list(range(0, 60))
    cal_idx = list(range(98, 196))

    st_last5 = raw_snapshot_stats(paths, last5_idx, rate)
    st_steady = raw_snapshot_stats(paths, steady_idx, rate)
    st_late = raw_snapshot_stats(paths, late_idx, rate)
    st_first60 = raw_snapshot_stats(paths, first60_idx, rate)
    st_cal = raw_snapshot_stats(paths, cal_idx, rate)

    print("\n  Reference raw medians [steady healthy 200..450] and [late 950..978]:")
    for ch in range(4):
        s_rms = np.median([st_steady[i][ch]["rms"] for i in steady_idx])
        s_ptp = np.median([st_steady[i][ch]["max"] - st_steady[i][ch]["min"] for i in steady_idx])
        l_rms = np.median([st_late[i][ch]["rms"] for i in late_idx])
        l_ptp = np.median([st_late[i][ch]["max"] - st_late[i][ch]["min"] for i in late_idx])
        print(f"    Ch {ch + 1}: steady_200_450 rms={s_rms:.4f}, ptp={s_ptp:.4f} | late_950_978 rms={l_rms:.4f}, ptp={l_ptp:.4f}")

    print("\n  Last 5 snapshots (979..983):")
    print(f"    {'Idx':>4} {'File':<22}{'Ch':>3}{'Len':>7}{'Min':>9}{'Max':>9}{'Mean':>9}{'Std':>9}{'RMS':>9}{'Kurt':>8}  Note")
    for idx in last5_idx:
        for ch in range(4):
            s = st_last5[idx][ch]
            note = "ANOMALOUS (near-zero)" if s["rms"] < 0.01 else ""
            print(
                f"    {idx:>4} {paths[idx].name:<22}{ch + 1:>3}{int(s['len']):>7}{s['min']:>9.4f}{s['max']:>9.4f}"
                f"{s['mean']:>9.4f}{s['std']:>9.4f}{s['rms']:>9.4f}{s['kurt']:>8.2f}  {note}"
            )

    print("\n  First 60 snapshots (0..59) vs calibration block (98..195):")
    for ch in range(4):
        cal_rms = np.array([st_cal[i][ch]["rms"] for i in cal_idx])
        cal_med, cal_sd = float(np.median(cal_rms)), float(np.std(cal_rms, ddof=0))
        f60_rms = np.array([st_first60[i][ch]["rms"] for i in first60_idx])
        f60_min = min(st_first60[i][ch]["min"] for i in first60_idx)
        f60_max = max(st_first60[i][ch]["max"] for i in first60_idx)
        f60_lens = {int(st_first60[i][ch]["len"]) for i in first60_idx}
        outliers = [i for i in first60_idx if abs(st_first60[i][ch]["rms"] - cal_med) > 3.0 * cal_sd]
        near_zero = [i for i in first60_idx if st_first60[i][ch]["rms"] < 0.01]
        print(
            f"    Ch {ch + 1}: lens={sorted(f60_lens)}, first60 rms=[{f60_rms.min():.4f}, {f60_rms.max():.4f}] "
            f"(med={np.median(f60_rms):.4f}), val_range=[{f60_min:.4f}, {f60_max:.4f}] | "
            f"cal rms med={cal_med:.4f} (sd={cal_sd:.4f}) | near_zero={near_zero} | "
            f"|rms-cal_med|>3sd count={len(outliers)} {outliers[:12]}"
        )
        print(f"           snap 0: rms={st_first60[0][ch]['rms']:.4f}, min={st_first60[0][ch]['min']:.4f}, max={st_first60[0][ch]['max']:.4f}, mean={st_first60[0][ch]['mean']:.4f}")

    # Recompute pipeline without last snapshot (0..982)
    t_no_last = t_hours[:-1]
    results_no_last = {
        key: evaluate_bearing(matrices[key][:-1], t_no_last, cfg, alpha=alpha, baseline_frac=base_frac)
        for key in BEARING_KEYS
    }
    p_with = root / "outputs/plots/ims_stage5b_all4_hi_with_last.png"
    p_without = root / "outputs/plots/ims_stage5b_all4_hi_without_last.png"
    plot_all4(results, t_hours, p_with, "(With last snapshot 0..983)")
    plot_all4(results_no_last, t_no_last, p_without, "(Without last snapshot 0..982)")
    print(f"\n  Plotted all 4 channels with last snapshot   : {p_with}")
    print(f"  Plotted all 4 channels without last snapshot: {p_without}")
    print("  End-of-test HI impact of dropping snapshot 983 (idx 982 vs 983):")
    for key in BEARING_KEYS:
        rw, rwo = results[key], results_no_last[key]
        print(
            f"    {key}: with_last end_HI={rw['end_hi_ema']:.4f} (raw[983]={rw['hi_raw'][-1]:.4f}, max={rw['max_hi_ema']:.4f}, lead_fix={rw['lead_fixed_h']:.2f}h) | "
            f"without_last end_HI={rwo['end_hi_ema']:.4f} (raw[982]={rwo['hi_raw'][-1]:.4f}, max={rwo['max_hi_ema']:.4f}, lead_fix={rwo['lead_fixed_h']:.2f}h)"
        )

    # ── ITEM 5 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 108}")
    print("ITEM 5: EMA INITIALIZATION & WINDOWED SPEARMAN RHO (196..983 and 525..983)")
    print("=" * 108)
    print("  EMA initialization check (hi_ema[0] == hi_raw[0]):")
    for key in BEARING_KEYS:
        r = results[key]
        print(f"    {key}: hi_raw[0]={r['hi_raw'][0]:.6f}, hi_ema[0]={r['hi_ema'][0]:.6f}, diff={abs(r['hi_ema'][0]-r['hi_raw'][0]):.2e}")
    print(
        f"\n  {'Bearing':<28}{'196..983 EMA rho':>18}{'196..983 Raw rho':>18}"
        f"{'525..983 EMA rho':>18}{'525..983 Raw rho':>18}{'525..982 EMA rho':>18}"
    )
    print("  " + "-" * 116)
    for key in BEARING_KEYS:
        r = results[key]
        re1, _ = spearmanr(t_hours[196:], r["hi_ema"][196:])
        rr1, _ = spearmanr(t_hours[196:], r["hi_raw"][196:])
        re2, _ = spearmanr(t_hours[525:], r["hi_ema"][525:])
        rr2, _ = spearmanr(t_hours[525:], r["hi_raw"][525:])
        re2_wo, _ = spearmanr(t_hours[525:-1], r["hi_ema"][525:-1])
        print(f"  {BEARING_LABELS[key]:<28}{re1:>18.4f}{rr1:>18.4f}{re2:>18.4f}{rr2:>18.4f}{re2_wo:>18.4f}")

    # ── ITEM 6 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 108}")
    print("ITEM 6 [EXPLORATORY / POST-HOC — NOT PART OF PRIMARY METHOD]: CROSS-BEARING RELATIVE Z-SCORES")
    print("=" * 108)
    print("  Definition: diff_b[t] = hi_ema[b, t] - median(hi_ema[others, t]); "
          "z_rel[b, t] = (diff_b[t] - mean(diff_b[:98])) / max(std(diff_b[:98], ddof=0), sigma_floor)")
    floor = float(cfg["hi_sigma_floor"])
    hi_mat = np.vstack([results[k]["hi_ema"] for k in BEARING_KEYS])  # (4, N)
    print(
        f"\n  {'Bearing':<28}{'diff@450':>9}{'diff@600':>9}{'diff@800':>9}{'diff@950':>9}"
        f"{'z_rel@450':>10}{'z_rel@600':>10}{'z_rel@800':>10}{'z_rel@950':>10}"
        f"{'med_z(525+)':>12}{'max_z(525+)':>12}{'frac>3(525+)':>13}"
    )
    print("  " + "-" * 136)
    for b_idx, key in enumerate(BEARING_KEYS):
        others = np.delete(hi_mat, b_idx, axis=0)
        diff = hi_mat[b_idx] - np.median(others, axis=0)
        mu_d = float(diff[:98].mean())
        sd_d = max(float(diff[:98].std(ddof=0)), floor)
        z_rel = (diff - mu_d) / sd_d
        z_late = z_rel[525:]
        print(
            f"  {BEARING_LABELS[key]:<28}{diff[450]:>9.3f}{diff[600]:>9.3f}{diff[800]:>9.3f}{diff[950]:>9.3f}"
            f"{z_rel[450]:>10.2f}{z_rel[600]:>10.2f}{z_rel[800]:>10.2f}{z_rel[950]:>10.2f}"
            f"{float(np.median(z_late)):>12.2f}{float(z_late.max()):>12.2f}{float(np.mean(z_late > 3.0)):>13.3f}"
        )


if __name__ == "__main__":
    main()

