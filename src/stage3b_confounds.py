"""Stage 3b: confound checks for the CWRU health index.

HI (one-sided):  z_f = max((x_f - mu_f) / max(sigma_f, floor), 0)
                 HI  = sqrt(mean_f(min(z_f, Z_CAP)^2))
Features: rms, peak_to_peak, crest_factor, kurtosis, log(band energies).
          skewness excluded, std dropped.  mu/sigma from first 70% of normal_0 only.
Threshold: mean(holdout HI) + k*std(holdout HI), holdout = normal_0 after a 1-window gap.
Nothing is tuned on fault windows.  Parameters come from config.yaml.

Variants evaluated
  3200 Hz : direct   (normal 48k->3200 in one resample_poly)   [current Stage 2/3 chain]
            twostep  (normal 48k->12k decimate -> 3200 resample_poly(4,15); same final
                      stage as the faults)
  12 kHz  : asis     (6 bands; normal decimated, faults native)
            drop3k   (drop the 3000-6000 Hz band)
            lpf      (faults zero-phase filtered with the decimate-filter magnitude response)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, firwin2, freqz

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (  # noqa: E402
    RECORDINGS, de_signal_and_metadata, extract_features, load_config, resample_to, window_signal,
)
from stage3_hi import auroc, load_rows, split_normal_0  # noqa: E402

EPS = np.finfo(float).eps
NORMAL_LABELS = ["normal", "normal_1", "normal_2", "normal_3"]
EXTRA_NORMAL = ["normal_1", "normal_2", "normal_3"]
FILE_OF = {label: name for name, label in RECORDINGS}
ONE_SIDED_BASE = ["rms", "peak_to_peak", "crest_factor", "kurtosis"]  # + all band energies
CFG: dict[str, Any] = {}
ROOT = Path(".")


# ── feature preparation ───────────────────────────────────────────────────────

def hi_feature_names(raw_cols: list[str], drop_bands: tuple[str, ...] = ()) -> list[str]:
    names = []
    for c in raw_cols:
        if c in ONE_SIDED_BASE:
            names.append(c)
        elif c.startswith("band_energy_") and c not in drop_bands:
            names.append("log_" + c)
    return names


def add_logs(rows: list[dict]) -> None:
    for r in rows:
        for k in [k for k in r if k.startswith("band_energy_")]:
            r["log_" + k] = float(np.log(max(r[k], EPS)))


def mat(rows: list[dict], feats: list[str]) -> np.ndarray:
    return np.array([[r[f] for f in feats] for r in rows], dtype=float)


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


# ── recomputation from raw for parity variants ────────────────────────────────

def decimate_magnitude_filter(fs_out: int, numtaps: int = 81) -> np.ndarray:
    """FIR at fs_out whose magnitude response equals scipy's decimate(q=4) FIR (designed at 48k)."""
    b48 = firwin(numtaps, 1.0 / 4, window="hamming")
    w, h = freqz(b48, 1, worN=2048, fs=48000)
    keep = w <= fs_out / 2
    freq, gain = w[keep], np.abs(h[keep])
    freq[-1] = fs_out / 2
    return firwin2(numtaps, freq, gain, fs=fs_out)


def rows_from_raw(label: str, rate: int, bands: list, chain: str) -> list[dict]:
    name = FILE_OF[label]
    sig, _ = de_signal_and_metadata(ROOT / "data/raw/cwru" / name)
    src = int(CFG["source_sampling_rates"][name])
    if chain == "twostep":                       # 48k -> 12k -> 3200
        sig = resample_to(resample_to(sig, src, 12000), 12000, rate)
    elif chain == "lpf":                         # native 12k, matched LPF, no resample
        b = decimate_magnitude_filter(rate)
        half = len(b) // 2
        sig = np.convolve(sig, b, mode="same")[half:-half]
    else:
        raise ValueError(chain)
    out = []
    for i, w in enumerate(window_signal(sig, int(CFG["window_size"]), float(CFG["overlap"]))):
        out.append({"file": name, "label": label, "window_idx": i, **extract_features(w, rate, bands)})
    return out


# ── evaluation of one variant ─────────────────────────────────────────────────

def evaluate(rows: list[dict], feats: list[str]) -> dict[str, Any]:
    z_cap, k = float(CFG["hi_z_cap"]), float(CFG["hi_threshold_k"])
    floor = float(CFG["hi_sigma_floor"])
    base, hold = split_normal_0(rows, float(CFG["baseline_fraction"]), gap=1)
    Xb = mat(base, feats)
    mu, sd = Xb.mean(0), np.maximum(Xb.std(0, ddof=0), floor)

    def score(rs):
        z = np.maximum((mat(rs, feats) - mu) / sd, 0.0)
        return np.sqrt(np.mean(np.minimum(z, z_cap) ** 2, 1)), np.sqrt(np.mean(z ** 2, 1)), np.mean(z >= z_cap, 1)

    for r in rows:
        r["hi"] = float(score([r])[0][0])
    hi_b, hi_h = score(base)[0], score(hold)[0]
    thr_mean, thr_std = float(hi_h.mean()), float(hi_h.std(ddof=1))
    thr = thr_mean + k * thr_std
    res: dict[str, Any] = dict(
        base=base, hold=hold, thr=thr, thr_mean=thr_mean, thr_std=thr_std, k=k, z_cap=z_cap, nfeat=len(feats),
        fpr_base=float(np.mean(hi_b > thr)), nb=len(base), nh=len(hold),
        fpr_hold=float(np.mean(hi_h > thr)), hold_hi=hi_h,
        normals={}, classes={},
    )
    pooled = []
    for lbl in EXTRA_NORMAL:
        rs = [r for r in rows if r["label"] == lbl]
        h = score(rs)[0]
        pooled.append(h)
        res["normals"][lbl] = dict(n=len(rs), med=float(np.median(h)), q25=pct(h, 25), q75=pct(h, 75),
                                   alert=float(np.mean(h > thr)), mx=float(h.max()))
    pooled = np.concatenate(pooled)
    for lbl in sorted({r["label"] for r in rows} - set(NORMAL_LABELS)):
        rs = [r for r in rows if r["label"] == lbl]
        h, hu, capf = score(rs)
        res["classes"][lbl] = dict(
            n=len(rs), auroc=auroc(hi_h, h), auroc13=auroc(pooled, h), recall=float(np.mean(h > thr)),
            med=float(np.median(h)), q25=pct(h, 25), q75=pct(h, 75), med_unc=float(np.median(hu)),
            frac_cap=float(np.mean(capf)),
        )
    return res


def show(title: str, r: dict, detail: bool = True) -> None:
    print(f"\n{'-'*100}\n{title}\n{'-'*100}")
    print(f"  features={r['nfeat']}  Z_CAP={r['z_cap']:.0f}  k={r['k']:.0f}  sigma_floor={CFG['hi_sigma_floor']}")
    print(f"  baseline n={r['nb']}  holdout n={r['nh']}  threshold = {r['thr_mean']:.4f} + {r['k']:.0f}x{r['thr_std']:.4f} = {r['thr']:.4f}")
    if r["nh"] < 15:
        print(f"  NOTE: holdout n={r['nh']} is too small for a conclusive FPR/threshold estimate.")
    print(f"  FPR on baseline-fit windows (in-sample): {r['fpr_base']:.3f} ({round(r['fpr_base']*r['nb'])}/{r['nb']})")
    print(f"  FPR on held-out normal_0               : {r['fpr_hold']:.3f} ({round(r['fpr_hold']*r['nh'])}/{r['nh']})")
    print(f"  {'normal file':<12}{'n':>5}{'medHI':>8}{'IQR':>8}{'maxHI':>8}{'alert rate':>12}")
    for lbl, d in r["normals"].items():
        print(f"  {lbl:<12}{d['n']:>5}{d['med']:>8.3f}{d['q75']-d['q25']:>8.3f}{d['mx']:>8.3f}{d['alert']:>12.3f}")
    print(f"  {'class':<16}{'n':>5}{'AUROC':>8}{'AUROC13':>9}{'recall':>8}{'medHI':>8}{'IQR':>7}{'medUncap':>10}{'frac@cap':>10}")
    for lbl, d in r["classes"].items():
        flag = "  <- weak" if d["auroc"] < 0.9 or d["recall"] < 0.5 else ""
        print(f"  {lbl:<16}{d['n']:>5}{d['auroc']:>8.4f}{d['auroc13']:>9.4f}{d['recall']:>8.3f}{d['med']:>8.3f}"
              f"{d['q75']-d['q25']:>7.3f}{d['med_unc']:>10.3f}{d['frac_cap']:>10.3f}{flag}")
    print("  (AUROC = vs held-out normal_0; AUROC13 = vs pooled normal_1-3; frac@cap = mean over windows & features of z>=cap)")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(rows: list[dict], r: dict, rate: int, path: Path) -> None:
    thr = r["thr"]
    base_idx = {x["window_idx"] for x in r["base"]}
    hold_idx = {x["window_idx"] for x in r["hold"]}
    labels = ["normal"] + EXTRA_NORMAL + sorted(r["classes"])
    xmax = max(x["window_idx"] for x in rows)
    fig, axes = plt.subplots(len(labels), 1, figsize=(14, 1.5 * len(labels)), sharex=True)
    for ax, lbl in zip(axes, labels):
        rs = sorted([x for x in rows if x["label"] == lbl], key=lambda x: x["window_idx"])
        x = np.array([q["window_idx"] for q in rs])
        y = np.array([q["hi"] for q in rs])
        if lbl == "normal":
            m_b = np.isin(x, list(base_idx))
            m_h = np.isin(x, list(hold_idx))
            ax.scatter(x[m_b], y[m_b], s=6, c="#888888", label="baseline fit")
            ax.scatter(x[m_h], y[m_h], s=6, c="k", label="holdout")
            ax.scatter(x[~m_b & ~m_h], y[~m_b & ~m_h], s=6, c="tab:orange", label="gap")
            ax.legend(fontsize=6, loc="upper right", ncol=3)
        else:
            ax.plot(x, y, lw=0.8, color="tab:blue" if lbl.startswith("normal") else "tab:green")
        ax.axhline(thr, color="red", ls="--", lw=0.8)
        ax.set_ylabel(lbl, rotation=0, ha="right", va="center", fontsize=7)
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("true window index within file (shared axis; shorter files simply end earlier)")
    fig.suptitle(f"One-sided HI per window @ {rate} Hz, threshold={thr:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global CFG
    CFG = load_config(Path("config.yaml"))
    plots = Path("outputs/plots")
    plots.mkdir(parents=True, exist_ok=True)

    # ---- 3200 Hz ----
    rows3, raw3 = load_rows(Path("data/processed/features_3200.csv"))
    add_logs(rows3)
    f3 = hi_feature_names(raw3)
    print("=" * 100 + "\nSTAGE 3b  (one-sided z, sigma floor, cap)\n" + "=" * 100)
    print(f"HI features @3200: {f3}")
    r_direct = evaluate(rows3, f3)
    show("3200 Hz  DIRECT chain (normal 48k->3200 single resample_poly(2,30))", r_direct)
    plot(rows3, r_direct, 3200, plots / "hi_per_window_3200hz.png")

    bands3 = CFG["fft_bands_hz"]
    rows3b = [r for r in rows3 if r["label"] not in NORMAL_LABELS]
    for lbl in NORMAL_LABELS:
        nr = rows_from_raw(lbl, 3200, bands3, "twostep")
        add_logs(nr)
        rows3b += nr
    r_two = evaluate(rows3b, f3)
    show("3200 Hz  TWO-STEP normal chain (48k->12k decimate -> 3200 resample_poly(4,15); same final stage as faults)", r_two)
    # normal_0 feature-level comparison (all windows)
    d0 = [r for r in rows3 if r["label"] == "normal"]
    t0 = [r for r in rows3b if r["label"] == "normal"]
    print("\n  normal_0 per-feature mean (direct vs two-step, all windows):")
    for f in f3:
        a, b = np.mean([r[f] for r in d0]), np.mean([r[f] for r in t0])
        print(f"    {f:<30} direct={a:>10.4f} twostep={b:>10.4f} diff={b-a:>+9.4f}")
    print(f"  normal_0 windows direct={len(d0)} twostep={len(t0)}")
    print("\n  3200 Hz direct vs two-step, delta AUROC (two-step minus direct) and median HI:")
    for lbl in r_direct["classes"]:
        a, b = r_direct["classes"][lbl], r_two["classes"][lbl]
        print(f"    {lbl:<16} AUROC {a['auroc']:.4f}->{b['auroc']:.4f}   medHI {a['med']:.3f}->{b['med']:.3f}   recall {a['recall']:.3f}->{b['recall']:.3f}")
    print(f"    threshold {r_direct['thr']:.4f}->{r_two['thr']:.4f}")
    plot(rows3b, r_two, 3200, plots / "hi_per_window_3200hz_twostep.png")

    # ---- 12 kHz ----
    rows12, raw12 = load_rows(Path("data/processed/features_12000.csv"))
    add_logs(rows12)
    f12 = hi_feature_names(raw12)
    print(f"\nHI features @12000: {f12}")
    r12 = evaluate(rows12, f12)
    show("12 kHz  AS-IS (6 bands; normal decimated by FIR 48k->12k, faults native 12k)", r12)
    plot(rows12, r12, 12000, plots / "hi_per_window_12000hz.png")

    f12d = hi_feature_names(raw12, drop_bands=("band_energy_3000_6000hz",))
    r12d = evaluate(rows12, f12d)
    show("12 kHz  DROP 3000-6000 Hz band (5 bands)", r12d)

    bands12 = CFG["fft_bands_hz_12k"]
    rows12l = [r for r in rows12 if r["label"] in NORMAL_LABELS]
    for lbl in sorted({r["label"] for r in rows12} - set(NORMAL_LABELS)):
        nr = rows_from_raw(lbl, 12000, bands12, "lpf")
        add_logs(nr)
        rows12l += nr
    r12l = evaluate(rows12l, f12)
    show("12 kHz  MATCHED LPF on faults (decimate-filter magnitude response, zero phase), 6 bands", r12l)

    print("\n  12 kHz AUROC / recall summary (as-is | drop3k | faults-LPF):")
    for lbl in r12["classes"]:
        a, b, c = r12["classes"][lbl], r12d["classes"][lbl], r12l["classes"][lbl]
        print(f"    {lbl:<16} AUROC {a['auroc']:.4f} | {b['auroc']:.4f} | {c['auroc']:.4f}    "
              f"recall {a['recall']:.3f} | {b['recall']:.3f} | {c['recall']:.3f}    medHI {a['med']:.2f} | {b['med']:.2f} | {c['med']:.2f}")
    print(f"    threshold         {r12['thr']:.4f} | {r12d['thr']:.4f} | {r12l['thr']:.4f}")
    print(f"    normal_1-3 alert  " + " | ".join(
        "/".join(f"{v['alert']:.2f}" for v in rr["normals"].values()) for rr in (r12, r12d, r12l)))
    plot(rows12l, r12l, 12000, plots / "hi_per_window_12000hz_lpf.png")


if __name__ == "__main__":
    main()

