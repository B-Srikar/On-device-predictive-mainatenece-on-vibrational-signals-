"""Stage 8: Simulated sensor resolution & noise sweep + baseline float16/int16 check.

All quantization and additive white Gaussian noise are SIMULATED on stored recordings,
not measurements from a physical accelerometer:
  - CWRU: noise & uniform mid-tread quantization applied ON THE 3200 Hz RESAMPLED STREAM
    (from data/processed/cwru_3200_conformance.bin) before feature extraction.
  - IMS : noise & uniform mid-tread quantization applied ON THE NATIVE 20,480 Hz WAVEFORMS
    (from data/processed/ims_ch1_waveforms.bin) before 39-window median extraction.
For cells with noise > 0, sweeps are repeated across `stage8_num_seeds` (10) seeds
and reported as mean ± std. Baseline mean, sigma, and thresholds are refitted on
degraded healthy data at every setting. IMS units are multiples of baseline RMS (no mg).
"""
from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).parent))
from ims_trend import rolling_slopes  # noqa: E402
from preprocess import load_config  # noqa: E402
from stage3_hi import auroc  # noqa: E402
from stage3c_lolo_log1p import NORMAL_LABELS, UNSEEN_ORDER  # noqa: E402
from stage5_ims_trend import consecutive_runs  # noqa: E402

EPSILON = np.finfo(float).eps


def extract_features_batch(W: np.ndarray, rate: int, bands: list[list[float]], band_masks: np.ndarray | None = None) -> np.ndarray:
    """Exact batch equivalent of preprocess.extract_features + log band energies."""
    W = np.asarray(W, dtype=np.float64)
    n = float(W.shape[1])
    Wc = W - W.mean(axis=1, keepdims=True)
    sq = Wc * Wc
    m2 = np.mean(sq, axis=1)
    rms = np.sqrt(m2)
    ptp = np.ptp(Wc, axis=1)
    abs_peak = np.max(np.abs(Wc), axis=1)
    crest = np.where(rms <= EPSILON, 0.0, abs_peak / np.maximum(rms, EPSILON))

    m4 = np.mean(sq * sq, axis=1)
    safe_m2_sq = np.maximum(m2 * m2, EPSILON)
    kurt_raw = ((n * n - 1.0) * (m4 / safe_m2_sq) - 3.0 * (n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0)) + 3.0
    kurt = np.where(m2 <= EPSILON, 0.0, kurt_raw)

    power = np.abs(np.fft.rfft(Wc, axis=1)) ** 2 / (n * n)
    if band_masks is None:
        freqs = np.fft.rfftfreq(int(n), d=1.0 / rate)
        band_masks = np.array(
            [(freqs >= lo) & (freqs <= hi if ib == len(bands) - 1 else freqs < hi) for ib, (lo, hi) in enumerate(bands)],
            dtype=bool,
        )
    # Matrix-multiply power (M, F) @ band_masks.T (F, B) -> (M, B)
    band_energies = power @ band_masks.T
    log_bands = np.log(np.maximum(band_energies, EPSILON))
    out = np.column_stack([rms, ptp, crest, kurt, log_bands])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def degrade_signal(x: np.ndarray, lsb: float, noise_rms: float, rng: np.random.Generator | None) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    if noise_rms > 0.0 and rng is not None:
        y += rng.normal(0.0, noise_rms, size=y.shape)
    if lsb > 0.0:
        y = np.round(y / lsb) * lsb
    return y


def score_log1p(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.maximum((X - mu) / sd, 0.0)
    return np.sqrt(np.mean(np.log1p(z) ** 2, axis=1))


def load_cwru_bin(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as h:
        (n_rec,) = struct.unpack("<I", h.read(4))
        labels, win_idx, samples, py_feats = [], [], [], []
        for _ in range(n_rec):
            lbl = h.read(32).split(b"\x00", 1)[0].decode("ascii")
            _fn = h.read(32)
            (widx,) = struct.unpack("<I", h.read(4))
            samp = np.frombuffer(h.read(1024 * 4), dtype=np.float32).copy()
            feats = np.frombuffer(h.read(8 * 4), dtype=np.float32).copy()
            _hi = h.read(4)
            labels.append(lbl)
            win_idx.append(widx)
            samples.append(samp)
            py_feats.append(feats)
    return labels, np.asarray(win_idx, dtype=int), np.vstack(samples), np.vstack(py_feats)


def load_ims_bins(wav_bin: Path, exp_bin: Path, all4_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with wav_bin.open("rb") as h:
        n_snaps, n_samp = struct.unpack("<II", h.read(8))
        wavs = np.frombuffer(h.read(n_snaps * n_samp * 4), dtype=np.float32).reshape((n_snaps, n_samp)).copy()
    with exp_bin.open("rb") as h:
        n_exp, _st, _ht, _hb = struct.unpack("<Ifff", h.read(16))
        assert n_exp == n_snaps
        py_feats = []
        for _ in range(n_exp):
            py_feats.append(np.frombuffer(h.read(9 * 4), dtype=np.float32).copy())
            _ = h.read(24)
    t_hours = []
    with all4_csv.open(newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r["bearing"] == "bearing_1" and int(r["snapshot_index"]) < n_snaps:
                t_hours.append((int(r["snapshot_index"]), float(r["elapsed_hours"])))
    t_hours.sort(key=lambda x: x[0])
    return wavs, np.asarray([x[1] for x in t_hours], dtype=float), np.vstack(py_feats)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size <= 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ── Part A: CWRU 3200 Hz Sweep ────────────────────────────────────────────────

def eval_cwru_lolo(X_all: np.ndarray, labels: list[str], win_idx: np.ndarray, cfg: dict[str, Any]) -> tuple[list[dict], dict]:
    frac = float(cfg["baseline_fraction"])
    k_thr = float(cfg["hi_threshold_k"])
    floor = float(cfg["hi_sigma_floor"])
    fault_labels = sorted(set(labels) - set(NORMAL_LABELS))
    lbl_arr = np.asarray(labels)

    per_class_records: list[dict] = []
    for unseen in UNSEEN_ORDER:
        seen = [l for l in NORMAL_LABELS if l != unseen]
        base_m = np.zeros(len(labels), dtype=bool)
        hold_m = np.zeros(len(labels), dtype=bool)
        for sl in seen:
            idxs = np.flatnonzero(lbl_arr == sl)
            idxs = idxs[np.argsort(win_idx[idxs])]
            split = int(len(idxs) * frac)
            base_m[idxs[:split]] = True
            hold_m[idxs[split + 1 :]] = True
        unseen_m = lbl_arr == unseen

        Xb = X_all[base_m]
        mu = Xb.mean(axis=0)
        sd = np.maximum(Xb.std(axis=0, ddof=0), floor)
        hi_h = score_log1p(X_all[hold_m], mu, sd)
        hi_u = score_log1p(X_all[unseen_m], mu, sd)
        hi_healthy = np.concatenate([hi_h, hi_u])
        thr = float(hi_h.mean() + k_thr * hi_h.std(ddof=1))
        thr_1pct = float(np.quantile(hi_healthy, 0.99))
        fpr_u = float(np.mean(hi_u > thr))

        for fl in fault_labels:
            hi_f = score_log1p(X_all[lbl_arr == fl], mu, sd)
            per_class_records.append(
                {
                    "unseen": unseen,
                    "fault_class": fl,
                    "auroc": auroc(hi_healthy, hi_f),
                    "rec_thr": float(np.mean(hi_f > thr)),
                    "rec_1pct": float(np.mean(hi_f > thr_1pct)),
                    "fpr_u": fpr_u,
                }
            )

    summary = {
        "mean_auroc": float(np.mean([r["auroc"] for r in per_class_records])),
        "min_auroc": float(min(r["auroc"] for r in per_class_records)),
        "mean_rec_thr": float(np.mean([r["rec_thr"] for r in per_class_records])),
        "mean_rec_1pct": float(np.mean([r["rec_1pct"] for r in per_class_records])),
        "mean_fpr_u": float(np.mean([r["fpr_u"] for r in per_class_records])),
        "max_fpr_u": float(max(r["fpr_u"] for r in per_class_records)),
    }
    return per_class_records, summary


def plot_cwru_heatmap(
    grid: dict[tuple[float, float], dict[str, Any]],
    lsbs: list[float],
    noises: list[float],
    num_seeds: int,
    out_path: Path,
) -> None:
    metrics = [
        ("mean_auroc", "4-Fold Mean AUROC (H+U)", "YlGn", (0.7, 1.0)),
        ("mean_rec_thr", "4-Fold Mean Recall @ Threshold", "YlGn", (0.2, 1.0)),
        ("mean_rec_1pct", "4-Fold Mean Recall @ 1% FPR", "YlGn", (0.2, 1.0)),
        ("mean_fpr_u", "4-Fold Mean Unseen-Load FPR", "OrRd", (0.0, 0.25)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    xlabels = ["Ref" if l == 0 else f"{l:g} mg" for l in lsbs]
    ylabels = [f"{n:g} mg" for n in noises]

    for ax, (key, title, cmap, (vmin, vmax)) in zip(axes.flat, metrics):
        arr = np.array([[grid[(l, n)][f"{key}_mean"] for l in lsbs] for n in noises], dtype=float)
        arr_sd = np.array([[grid[(l, n)][f"{key}_std"] for l in lsbs] for n in noises], dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(lsbs)), xlabels, fontsize=8.5)
        ax.set_yticks(range(len(noises)), ylabels, fontsize=8.5)
        ax.set_xlabel("Simulated ADC LSB size (on 3200 Hz stream)", fontsize=8.5)
        ax.set_ylabel("Simulated AWGN RMS", fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        mid = 0.5 * (vmin + vmax)
        for r in range(len(noises)):
            for c in range(len(lsbs)):
                val, sd = arr[r, c], arr_sd[r, c]
                col = "white" if (key != "mean_fpr_u" and val < mid) or (key == "mean_fpr_u" and val > mid) else "black"
                txt = f"{val:.3f}" if noises[r] == 0.0 else f"{val:.3f}\n±{sd:.3f}"
                ax.text(c, r, txt, ha="center", va="center", color=col, fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Stage 8 (Simulated Sensor Sweep on 3200 Hz Stream — NOT Measured Sensor): CWRU 4-Fold LOLO\n"
        f"(Baseline refitted per cell; noise>0 cells show mean ± std across {num_seeds} seeds)",
        fontsize=10.5,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Part B: IMS Sweep ─────────────────────────────────────────────────────────

def extract_ims_snapshot_medians(
    wavs: np.ndarray, rate: int, bands: list[list[float]], band_masks: np.ndarray
) -> np.ndarray:
    n_snaps = wavs.shape[0]
    meds = np.empty((n_snaps, 4 + len(bands)), dtype=np.float64)
    chunk = 200
    for s in range(0, n_snaps, chunk):
        e = min(n_snaps, s + chunk)
        wins = sliding_window_view(wavs[s:e], 1024, axis=1)[:, ::512, :]
        B = e - s
        feats = extract_features_batch(wins.reshape(B * 39, 1024), rate, bands, band_masks=band_masks)
        meds[s:e] = np.median(feats.reshape(B, 39, -1), axis=1)
    return meds


def eval_ims_metrics(X: np.ndarray, t_hours: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    n = len(X)
    n_base = int(n * float(cfg["ims_baseline_fraction"]))
    n_cal = int(n * float(cfg["ims_calibration_fraction"]))
    cal_end = n_base + n_cal
    floor = float(cfg["hi_sigma_floor"])
    alpha = float(cfg["ims_ema_alpha"])
    k_hi = float(cfg["ims_threshold_k"])
    w_slope = int(cfg["ims_slope_window"])
    k_slope = float(cfg["ims_slope_threshold_sigma"])
    m_consec = int(cfg["ims_consecutive_alarms"])
    off_ratio = float(cfg["ims_machine_off_rms_ratio"])
    hlth_lo, hlth_hi = [int(v) for v in cfg["ims_presumed_healthy_range"]]

    mu = X[:n_base].mean(axis=0)
    sd = np.maximum(X[:n_base].std(axis=0, ddof=0), floor)
    rms_off_thr = off_ratio * float(np.median(X[:n_base, 0]))
    gated = X[:, 0] < rms_off_thr

    hi_raw = score_log1p(X, mu, sd)
    hi_ema = np.empty_like(hi_raw)
    hi_ema[0] = hi_raw[0]
    for t in range(1, n):
        hi_ema[t] = hi_ema[t - 1] if gated[t] else (alpha * hi_raw[t] + (1.0 - alpha) * hi_ema[t - 1])

    slopes = rolling_slopes(hi_ema, w_slope)
    hi_thr = float(hi_ema[n_base:cal_end].mean() + k_hi * hi_ema[n_base:cal_end].std(ddof=1))
    cal_slp = slopes[n_base:cal_end]
    cal_slp = cal_slp[np.isfinite(cal_slp)]
    slp_thr = float(cal_slp.mean() + k_slope * cal_slp.std(ddof=1))
    base_mean = float(hi_ema[:n_base].mean())

    trig_f = consecutive_runs((hi_ema > hi_thr) & (~gated), m_consec)
    trig_s = consecutive_runs(np.isfinite(slopes) & (slopes > slp_thr) & (hi_ema > base_mean) & (~gated), m_consec)

    pf = np.flatnonzero(trig_f[cal_end:])
    ps = np.flatnonzero(trig_s[cal_end:])
    first_f = None if pf.size == 0 else int(cal_end + pf[0])
    first_s = None if ps.size == 0 else int(cal_end + ps[0])
    lead_f = float(t_hours[-1] - t_hours[first_f]) if first_f is not None else float("nan")
    lead_s = float(t_hours[-1] - t_hours[first_s]) if first_s is not None else float("nan")
    return {
        "first_fixed": first_f,
        "lead_fixed_h": lead_f,
        "first_slope": first_s,
        "lead_slope_h": lead_s,
        "hlth_fixed": int(trig_f[hlth_lo : hlth_hi + 1].sum()),
        "hlth_slope": int(trig_s[hlth_lo : hlth_hi + 1].sum()),
        "gated": int(gated.sum()),
    }


def plot_ims_heatmap(
    grid: dict[tuple[int, float], dict[str, Any]],
    divs: list[int],
    ratios: list[float],
    num_seeds: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.2))
    xlabels = ["Ref" if d == 0 else f"rms/{d}\n({1.0 / d:g}x)" for d in divs]
    ylabels = [f"{r:g} x rms" for r in ratios]

    specs = [
        ("lead_fixed_h", "first_fixed", "Fixed Alarm Lead Time (h) & idx", "YlGn", (40.0, 85.0)),
        ("lead_slope_h", "first_slope", "Slope Alarm Lead Time (h) & idx", "YlGn", (40.0, 85.0)),
        ("hlth_total", None, "Presumed-Healthy Alarms (196..491)", "OrRd", (0.0, 10.0)),
    ]
    for ax, (val_k, idx_k, title, cmap, (vmin, vmax)) in zip(axes, specs):
        if val_k == "hlth_total":
            arr = np.array(
                [[grid[(d, r)]["hlth_fixed_mean"] + grid[(d, r)]["hlth_slope_mean"] for d in divs] for r in ratios],
                dtype=float,
            )
        else:
            arr = np.array([[grid[(d, r)][f"{val_k}_mean"] for d in divs] for r in ratios], dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(divs)), xlabels, fontsize=8)
        ax.set_yticks(range(len(ratios)), ylabels, fontsize=8)
        ax.set_xlabel("Simulated ADC LSB (multiples of baseline RMS)", fontsize=8.5)
        ax.set_ylabel("Simulated AWGN RMS (multiples of baseline RMS)", fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        mid = 0.5 * (vmin + vmax)
        for ri, r in enumerate(ratios):
            for ci, d in enumerate(divs):
                g = grid[(d, r)]
                if val_k == "hlth_total":
                    if r == 0.0:
                        txt = f"F:{g['hlth_fixed_mean']:.0f} / S:{g['hlth_slope_mean']:.0f}"
                    else:
                        txt = f"F:{g['hlth_fixed_mean']:.1f}±{g['hlth_fixed_std']:.1f}\nS:{g['hlth_slope_mean']:.1f}±{g['hlth_slope_std']:.1f}"
                    col = "white" if arr[ri, ci] > mid else "black"
                else:
                    vm, vs = g[f"{val_k}_mean"], g[f"{val_k}_std"]
                    im_v = g[f"{idx_k}_mean"]
                    if r == 0.0:
                        txt = f"{vm:.2f}h\n@{im_v:.0f}"
                    else:
                        txt = f"{vm:.1f}±{vs:.1f}h\n@{im_v:.0f}"
                    col = "white" if (not np.isfinite(vm) or vm < mid) else "black"
                ax.text(ci, ri, txt, ha="center", va="center", color=col, fontsize=7.0, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Stage 8 (Simulated Sensor Sweep on Native 20,480 Hz Signal — NOT Measured Sensor): IMS Ch 1\n"
        f"(Multiples of baseline RMS; refitted on degraded signal; noise>0 cells show mean ± std across {num_seeds} seeds)",
        fontsize=10.5,
    )
    fig.tight_layout(w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Part C: float16 and int16 baseline parameter quantization ─────────────────

def quantize_int16_array(v: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(v)) / 32767.0)
    q = np.clip(np.round(v / scale), -32767, 32767).astype(np.int16)
    deq = (q.astype(np.float64) * scale).astype(np.float32)
    return deq, scale


def check_baseline_storage(
    X_cwru: np.ndarray, labels_cwru: list[str], win_idx_cwru: np.ndarray,
    X_ims: np.ndarray, cfg: dict[str, Any],
) -> None:
    floor = np.float32(cfg["hi_sigma_floor"])
    print(f"\n{'=' * 116}")
    print("PART C: BASELINE MEAN / SIGMA STORAGE AS FLOAT16 & INT16 FIXED POINT VS FLOAT32")
    print("=" * 116)

    lbl_arr = np.asarray(labels_cwru)
    base_m = np.zeros(len(labels_cwru), dtype=bool)
    hold_m = np.zeros(len(labels_cwru), dtype=bool)
    for sl in ("normal", "normal_1", "normal_2"):
        idxs = np.flatnonzero(lbl_arr == sl)
        idxs = idxs[np.argsort(win_idx_cwru[idxs])]
        sp = int(len(idxs) * float(cfg["baseline_fraction"]))
        base_m[idxs[:sp]] = True
        hold_m[idxs[sp + 1 :]] = True
    unseen_m = lbl_arr == "normal_3"

    Xb = X_cwru[base_m]
    mu_32 = Xb.mean(axis=0).astype(np.float32)
    sd_32 = np.maximum(Xb.std(axis=0, ddof=0).astype(np.float32), floor)
    mu_16 = mu_32.astype(np.float16).astype(np.float32)
    sd_16 = np.maximum(sd_32.astype(np.float16).astype(np.float32), floor)
    mu_i16, mu_sc = quantize_int16_array(mu_32)
    sd_i16_raw, sd_sc = quantize_int16_array(sd_32)
    sd_i16 = np.maximum(sd_i16_raw, floor)

    X32_cwru = X_cwru.astype(np.float32)
    hi_ref = score_log1p(X32_cwru, mu_32, sd_32)
    fault_labels = sorted(set(labels_cwru) - set(NORMAL_LABELS))
    hi_healthy_ref = np.concatenate([hi_ref[hold_m], hi_ref[unseen_m]])
    min_auc_ref = min(auroc(hi_healthy_ref, hi_ref[lbl_arr == f]) for f in fault_labels)

    print("  1) CWRU 3200 Hz (normal_0-2 baseline, F=8, all 774 windows):")
    print(f"     int16 scales: mu_scale={mu_sc:.3e}, sd_scale={sd_sc:.3e}")
    for tag, m_use, s_use, nbytes in (
        ("float16", mu_16, sd_16, 4 * 8),
        ("int16 (per-vec scale)", mu_i16, sd_i16, 4 * 8 + 8),
    ):
        hi_q = score_log1p(X32_cwru, m_use, s_use)
        diff = np.abs(hi_q - hi_ref)
        rel = diff / np.maximum(np.abs(hi_ref), 1e-12)
        thr_ref = float(hi_ref[hold_m].mean() + 3.0 * hi_ref[hold_m].std(ddof=1))
        thr_q = float(hi_q[hold_m].mean() + 3.0 * hi_q[hold_m].std(ddof=1))
        hi_h_q = np.concatenate([hi_q[hold_m], hi_q[unseen_m]])
        min_auc_q = min(auroc(hi_h_q, hi_q[lbl_arr == f]) for f in fault_labels)
        rec_diff = max(
            abs(float(np.mean(hi_ref[lbl_arr == f] > thr_ref)) - float(np.mean(hi_q[lbl_arr == f] > thr_q)))
            for f in fault_labels
        )
        print(
            f"     {tag:<22}: bytes={nbytes:2d}B (vs 64B f32) | HI max_abs={diff.max():.3e}, "
            f"med_abs={np.median(diff):.3e}, max_rel={rel.max():.3e} | "
            f"min_AUROC {min_auc_ref:.4f}->{min_auc_q:.4f}, max_rec_delta={rec_diff:.4f}"
        )

    n_ims = len(X_ims)
    n_b, n_c = int(n_ims * 0.10), int(n_ims * 0.10)
    Xb_i = X_ims[:n_b]
    mu_32 = Xb_i.mean(axis=0).astype(np.float32)
    sd_32 = np.maximum(Xb_i.std(axis=0, ddof=0).astype(np.float32), floor)
    mu_16 = mu_32.astype(np.float16).astype(np.float32)
    sd_16 = np.maximum(sd_32.astype(np.float16).astype(np.float32), floor)
    mu_i16, mu_sc = quantize_int16_array(mu_32)
    sd_i16_raw, sd_sc = quantize_int16_array(sd_32)
    sd_i16 = np.maximum(sd_i16_raw, floor)
    X32_ims = X_ims.astype(np.float32)

    def _run_ims_q(m_use: np.ndarray, s_use: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
        raw = score_log1p(X32_ims, m_use, s_use)
        ema_arr = np.empty_like(raw)
        ema_arr[0] = raw[0]
        for t in range(1, n_ims):
            ema_arr[t] = 0.1 * raw[t] + 0.9 * ema_arr[t - 1]
        slp = rolling_slopes(ema_arr, 12)
        ht = float(ema_arr[n_b : n_b + n_c].mean() + 3.0 * ema_arr[n_b : n_b + n_c].std(ddof=1))
        cs = slp[n_b : n_b + n_c]
        cs = cs[np.isfinite(cs)]
        st = float(cs.mean() + 3.0 * cs.std(ddof=1))
        tf = consecutive_runs(ema_arr > ht, 3)
        ts = consecutive_runs(np.isfinite(slp) & (slp > st) & (ema_arr > ema_arr[:n_b].mean()), 3)
        pf = np.flatnonzero(tf[n_b + n_c :])
        ps = np.flatnonzero(ts[n_b + n_c :])
        return raw, ema_arr, (None if pf.size == 0 else int(n_b + n_c + pf[0])), (None if ps.size == 0 else int(n_b + n_c + ps[0]))

    raw_ref, ema_ref, ff_ref, fs_ref = _run_ims_q(mu_32, sd_32)
    print("\n  2) IMS Ch 1 (snapshots 0..981, F=9):")
    print(f"     int16 scales: mu_scale={mu_sc:.3e}, sd_scale={sd_sc:.3e} | float32 ref: fixed={ff_ref}, slope={fs_ref}")
    for tag, m_use, s_use, nbytes in (
        ("float16", mu_16, sd_16, 4 * 9),
        ("int16 (per-vec scale)", mu_i16, sd_i16, 4 * 9 + 8),
    ):
        raw_q, ema_q, ff_q, fs_q = _run_ims_q(m_use, s_use)
        print(
            f"     {tag:<22}: bytes={nbytes:2d}B (vs 72B f32) | raw_HI max_abs={np.abs(raw_q - raw_ref).max():.3e}, "
            f"EMA max_abs={np.abs(ema_q - ema_ref).max():.3e}, med_abs={np.median(np.abs(ema_q - ema_ref)):.3e} | "
            f"alarms: fixed={ff_q}, slope={fs_q}"
        )


def main() -> None:
    root = Path(".")
    cfg = load_config(root / "config.yaml")
    seed_base = int(cfg["stage8_seed"])
    num_seeds = int(cfg["stage8_num_seeds"])

    labels_cwru, win_idx_cwru, samp_cwru, py_feats_cwru = load_cwru_bin(root / "data/processed/cwru_3200_conformance.bin")
    wavs_ims, t_ims, py_feats_ims = load_ims_bins(
        root / "data/processed/ims_ch1_waveforms.bin",
        root / "data/processed/ims_ch1_expected.bin",
        root / "data/processed/ims_2nd_test_all4_features.csv",
    )

    cwru_freqs = np.fft.rfftfreq(1024, d=1.0 / int(cfg["target_rate"]))
    cwru_masks = np.array(
        [(cwru_freqs >= lo) & (cwru_freqs <= hi if ib == len(cfg["fft_bands_hz"]) - 1 else cwru_freqs < hi)
         for ib, (lo, hi) in enumerate(cfg["fft_bands_hz"])],
        dtype=bool,
    )
    ims_freqs = np.fft.rfftfreq(1024, d=1.0 / int(cfg["ims_sampling_rate"]))
    ims_masks = np.array(
        [(ims_freqs >= lo) & (ims_freqs <= hi if ib == len(cfg["ims_fft_bands_hz"]) - 1 else ims_freqs < hi)
         for ib, (lo, hi) in enumerate(cfg["ims_fft_bands_hz"])],
        dtype=bool,
    )

    X_cwru_ref = extract_features_batch(samp_cwru, int(cfg["target_rate"]), cfg["fft_bands_hz"], cwru_masks)
    X_ims_ref = extract_ims_snapshot_medians(wavs_ims.astype(np.float64), int(cfg["ims_sampling_rate"]), cfg["ims_fft_bands_hz"], ims_masks)
    diff_c = float(np.max(np.abs(X_cwru_ref - py_feats_cwru)))
    diff_i = float(np.max(np.abs(X_ims_ref - py_feats_ims)))
    ims_ref_res = eval_ims_metrics(X_ims_ref, t_ims, cfg)
    print(f"Batch verifier: CWRU max_diff={diff_c:.2e}, IMS max_diff={diff_i:.2e}, "
          f"IMS unquantized alarms=({ims_ref_res['first_fixed']}, {ims_ref_res['first_slope']})")
    if diff_c > 1e-4 or diff_i > 1e-4 or (ims_ref_res["first_fixed"], ims_ref_res["first_slope"]) != (527, 538):
        raise RuntimeError("Batch extractor failed sanity parity check")

    # ── Part A: CWRU Sweep (on 3200 Hz resampled stream) ──────────────────────
    cwru_lsbs = [float(v) for v in cfg["stage8_cwru_lsb_mg"]]
    cwru_noises = [float(v) for v in cfg["stage8_cwru_noise_mg"]]
    cwru_grid: dict[tuple[float, float], dict[str, Any]] = {}
    cwru_class_rows: list[dict] = []

    print(f"\n{'=' * 116}")
    print(f"PART A: SIMULATED SENSOR SWEEP — CWRU 3200 Hz STREAM (4 LOLO folds, {num_seeds} seeds for noise > 0)")
    print("Application point: noise & quantization applied directly ON the 3200 Hz resampled stream (NOT raw 48k/12k, NOT measured sensor).")
    print("=" * 116)
    print(f"  {'LSB(mg)':>8}{'Noise(mg)':>10}{'MeanAUC':>16}{'MinAUC':>16}{'Rec@thr':>16}{'Rec@1%FPR':>16}{'FPR_U':>16}")
    print("  " + "-" * 98)

    for noise_mg in cwru_noises:
        seeds = [seed_base] if noise_mg == 0.0 else [seed_base + s for s in range(num_seeds)]
        for lsb_mg in cwru_lsbs:
            seed_summs: list[dict] = []
            seed_recs: list[list[dict]] = []
            for s in seeds:
                rng = np.random.default_rng(s)
                deg = degrade_signal(samp_cwru, lsb_mg * 1e-3, noise_mg * 1e-3, rng if noise_mg > 0 else None)
                X_deg = extract_features_batch(deg, int(cfg["target_rate"]), cfg["fft_bands_hz"], cwru_masks)
                recs, summ = eval_cwru_lolo(X_deg, labels_cwru, win_idx_cwru, cfg)
                seed_summs.append(summ)
                seed_recs.append(recs)

            cell: dict[str, Any] = {"n_seeds": len(seeds)}
            for mk in ("mean_auroc", "min_auroc", "mean_rec_thr", "mean_rec_1pct", "mean_fpr_u"):
                m, sd = _mean_std([x[mk] for x in seed_summs])
                cell[f"{mk}_mean"] = m
                cell[f"{mk}_std"] = sd
            cwru_grid[(lsb_mg, noise_mg)] = cell

            l_str = "Ref" if lsb_mg == 0 else f"{lsb_mg:g}"
            print(
                f"  {l_str:>8}{noise_mg:>10.1f}"
                f"  {cell['mean_auroc_mean']:.4f}±{cell['mean_auroc_std']:.4f}"
                f"  {cell['min_auroc_mean']:.4f}±{cell['min_auroc_std']:.4f}"
                f"   {cell['mean_rec_thr_mean']:.3f}±{cell['mean_rec_thr_std']:.3f}"
                f"   {cell['mean_rec_1pct_mean']:.3f}±{cell['mean_rec_1pct_std']:.3f}"
                f"   {cell['mean_fpr_u_mean']:.3f}±{cell['mean_fpr_u_std']:.3f}"
            )

            # Aggregate per-class mean across 4 folds per seed, then mean/std across seeds
            for fl in sorted(set(labels_cwru) - set(NORMAL_LABELS)):
                au_s = [float(np.mean([r["auroc"] for r in sr if r["fault_class"] == fl])) for sr in seed_recs]
                rt_s = [float(np.mean([r["rec_thr"] for r in sr if r["fault_class"] == fl])) for sr in seed_recs]
                r1_s = [float(np.mean([r["rec_1pct"] for r in sr if r["fault_class"] == fl])) for sr in seed_recs]
                am, asd = _mean_std(au_s)
                tm, tsd = _mean_std(rt_s)
                om, osd = _mean_std(r1_s)
                cwru_class_rows.append(
                    {
                        "lsb_mg": lsb_mg,
                        "noise_mg": noise_mg,
                        "n_seeds": len(seeds),
                        "fault_class": fl,
                        "auroc_mean": am,
                        "auroc_std": asd,
                        "rec_thr_mean": tm,
                        "rec_thr_std": tsd,
                        "rec_1pct_mean": om,
                        "rec_1pct_std": osd,
                    }
                )

    key_corners = [(0.0, 0.0), (15.6, 5.0), (31.2, 0.0), (31.2, 10.0)]
    print("\n  Per-class 4-fold mean across 10 seeds [AUROC / Rec@thr / Rec@1%FPR]:")
    print(f"  {'Class':<16}" + "".join(f"{('Ref' if l==0 else f'{l:g}mg') + f'+{n:g}mg':>24}" for l, n in key_corners))
    print("  " + "-" * (16 + 24 * len(key_corners)))
    for fl in sorted(set(labels_cwru) - set(NORMAL_LABELS)):
        cells = []
        for l, n in key_corners:
            r = next(x for x in cwru_class_rows if x["lsb_mg"] == l and x["noise_mg"] == n and x["fault_class"] == fl)
            cells.append(f"{r['auroc_mean']:.3f}/{r['rec_thr_mean']:.2f}/{r['rec_1pct_mean']:.2f}")
        print(f"  {fl:<16}" + "".join(f"{c:>24}" for c in cells))

    cwru_csv = root / "outputs/stage8_cwru_sweep.csv"
    with cwru_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(cwru_class_rows[0].keys()))
        w.writeheader()
        w.writerows(cwru_class_rows)
    plot_cwru_heatmap(cwru_grid, cwru_lsbs, cwru_noises, num_seeds, root / "outputs/plots/stage8_cwru_heatmap.png")

    # ── Part B: IMS Sweep (multiples of baseline RMS, NO mg) ──────────────────
    ims_divs = [int(v) for v in cfg["stage8_ims_lsb_divisors"]]
    ims_ratios = [float(v) for v in cfg["stage8_ims_noise_ratios"]]
    base_rms_ref = float(np.median(X_ims_ref[: int(len(X_ims_ref) * 0.10), 0]))
    ims_rows_out: list[dict] = []
    ims_grid: dict[tuple[int, float], dict[str, Any]] = {}

    print(f"\n{'=' * 116}")
    print(f"PART B: SIMULATED SENSOR SWEEP — IMS Ch 1 NATIVE 20,480 Hz ({num_seeds} seeds for noise > 0)")
    print("Application point: noise & quantization applied on native 20,480 Hz waveforms; units are multiples of baseline RMS.")
    print("=" * 116)
    print(
        f"  {'LSB (x rms)':>16}{'Noise (x rms)':>15}{'Fixed idx (m±sd)':>18}{'Fixed lead (h)':>18}"
        f"{'Slope idx (m±sd)':>18}{'Slope lead (h)':>18}{'196..491 (F / S)':>18}{'Gated':>7}"
    )
    print("  " + "-" * 126)

    for nr in ims_ratios:
        seeds = [seed_base] if nr == 0.0 else [seed_base + s for s in range(num_seeds)]
        for div in ims_divs:
            lsb_val = 0.0 if div == 0 else base_rms_ref / float(div)
            noise_val = base_rms_ref * nr
            seed_res: list[dict[str, Any]] = []
            for s in seeds:
                rng = np.random.default_rng(s)
                deg_wavs = degrade_signal(wavs_ims, lsb_val, noise_val, rng if nr > 0 else None)
                X_deg_ims = extract_ims_snapshot_medians(
                    deg_wavs, int(cfg["ims_sampling_rate"]), cfg["ims_fft_bands_hz"], ims_masks
                )
                seed_res.append(eval_ims_metrics(X_deg_ims, t_ims, cfg))

            cell: dict[str, Any] = {"n_seeds": len(seeds)}
            for mk in ("first_fixed", "lead_fixed_h", "first_slope", "lead_slope_h", "hlth_fixed", "hlth_slope"):
                m, sd = _mean_std([float(x[mk]) for x in seed_res])
                cell[f"{mk}_mean"] = m
                cell[f"{mk}_std"] = sd
            cell["gated_max"] = max(int(x["gated"]) for x in seed_res)
            ims_grid[(div, nr)] = cell

            l_lbl = "Ref" if div == 0 else f"rms/{div} ({1.0/div:g}x)"
            n_lbl = f"{nr:g} x rms"
            print(
                f"  {l_lbl:>16}{n_lbl:>15}"
                f"  {cell['first_fixed_mean']:6.1f}±{cell['first_fixed_std']:4.1f}    "
                f"  {cell['lead_fixed_h_mean']:6.2f}±{cell['lead_fixed_h_std']:4.2f}    "
                f"  {cell['first_slope_mean']:6.1f}±{cell['first_slope_std']:4.1f}    "
                f"  {cell['lead_slope_h_mean']:6.2f}±{cell['lead_slope_h_std']:4.2f}    "
                f"{cell['hlth_fixed_mean']:5.1f} / {cell['hlth_slope_mean']:4.1f}    "
                f"{cell['gated_max']:>7}"
            )
            ims_rows_out.append(
                {
                    "lsb_divisor": div,
                    "lsb_mult_rms": 0.0 if div == 0 else 1.0 / div,
                    "noise_mult_rms": nr,
                    "n_seeds": len(seeds),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in cell.items()},
                }
            )

    ims_csv = root / "outputs/stage8_ims_sweep.csv"
    with ims_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(ims_rows_out[0].keys()))
        w.writeheader()
        w.writerows(ims_rows_out)
    plot_ims_heatmap(ims_grid, ims_divs, ims_ratios, num_seeds, root / "outputs/plots/stage8_ims_heatmap.png")

    check_baseline_storage(X_cwru_ref, labels_cwru, win_idx_cwru, X_ims_ref, cfg)
    print(f"\nSaved CSVs: {cwru_csv}, {ims_csv}")
    print("Saved Heatmaps: outputs/plots/stage8_cwru_heatmap.png, outputs/plots/stage8_ims_heatmap.png")


if __name__ == "__main__":
    main()
