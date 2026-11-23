"""Export binary conformance datasets for Stage 6b C harnesses:
  1. data/processed/cwru_3200_conformance.bin (all 774 windows in features_3200.csv)
  2. data/processed/ims_ch1_waveforms.bin and data/processed/ims_ch1_expected.bin
     (IMS Set No. 2 Ch 1, snapshots 0..981 excluding 982-983).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]

from export_baseline_header import compute_ims_ch1_baseline, compute_pooled_normal_0_2_baseline  # noqa: E402
from ims_trend import load_channel, snapshot_paths  # noqa: E402
from preprocess import (  # noqa: E402
    RECORDINGS,
    de_signal_and_metadata,
    extract_features,
    load_config,
    matched_rate_signal,
    window_signal,
)
from stage3_hi import load_rows  # noqa: E402
from stage3b_confounds import add_logs  # noqa: E402
from stage5_ims_trend import consecutive_runs  # noqa: E402

EPSILON = np.finfo(float).eps


def _pack_fixed_str(text: str, length: int = 32) -> bytes:
    raw = text.encode("ascii")[: length - 1]
    return raw + b"\x00" * (length - len(raw))


def export_cwru_all774(config_path: Path) -> None:
    cfg = load_config(config_path)
    feat_csv = ROOT / "data/processed/features_3200.csv"
    stats = compute_pooled_normal_0_2_baseline(config_path, feat_csv)
    mean: np.ndarray = stats["mean"]  # type: ignore[assignment]
    std: np.ndarray = stats["std"]  # type: ignore[assignment]
    feats: list[str] = stats["feats"]  # type: ignore[assignment]

    csv_rows, _ = load_rows(feat_csv)
    add_logs(csv_rows)
    csv_lookup = {(r["file"], int(r["window_idx"])): np.array([r[f] for f in feats], dtype=float) for r in csv_rows}

    base_rate = int(cfg["sampling_rate"])
    target_rate = int(cfg["target_rate"])
    win_size = int(cfg["window_size"])
    overlap = float(cfg["overlap"])
    bands = cfg["fft_bands_hz"]
    source_rates = cfg["source_sampling_rates"]
    raw_band_cols = [f"band_energy_{int(lo)}_{int(hi)}hz" for lo, hi in bands]

    records: list[tuple[str, str, int, np.ndarray, np.ndarray, float]] = []
    max_csv_diff = 0.0
    for filename, label in RECORDINGS:
        raw_sig, _ = de_signal_and_metadata(ROOT / "data/raw/cwru" / filename)
        sig, rate = matched_rate_signal(raw_sig, filename, label, int(source_rates[filename]), base_rate, target_rate)
        for idx, win in enumerate(window_signal(sig, win_size, overlap)):
            rf = extract_features(win, rate, bands)
            fvec = np.array(
                [rf["rms"], rf["peak_to_peak"], rf["crest_factor"], rf["kurtosis"]]
                + [float(np.log(max(rf[c], EPSILON))) for c in raw_band_cols],
                dtype=float,
            )
            max_csv_diff = max(max_csv_diff, float(np.max(np.abs(fvec - csv_lookup[(filename, idx)]))))
            z = np.maximum((fvec - mean) / std, 0.0)
            hi = float(np.sqrt(np.mean(np.log1p(z) ** 2)))
            records.append((label, filename, idx, win.astype(np.float32), fvec.astype(np.float32), hi))

    if len(records) != len(csv_rows):
        raise RuntimeError(f"Window count mismatch: {len(records)} vs {len(csv_rows)}")
    if max_csv_diff > 1e-12:
        raise RuntimeError(f"Feature mismatch vs features_3200.csv: {max_csv_diff}")

    out_bin = ROOT / "data/processed/cwru_3200_conformance.bin"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as handle:
        handle.write(struct.pack("<I", len(records)))
        for label, filename, idx, win_f32, feat_f32, hi in records:
            handle.write(_pack_fixed_str(label, 32))
            handle.write(_pack_fixed_str(filename, 32))
            handle.write(struct.pack("<I", idx))
            handle.write(win_f32.tobytes())
            handle.write(feat_f32.tobytes())
            handle.write(struct.pack("<f", hi))
    print(f"Wrote {len(records)} CWRU 3200 Hz windows (csv diff={max_csv_diff:.2e}) to {out_bin}")


def export_ims_ch1(config_path: Path) -> None:
    cfg = load_config(config_path)
    stats = compute_ims_ch1_baseline(config_path, ROOT / "data/processed/ims_2nd_test_all4_features.csv")
    X: np.ndarray = stats["X"]  # type: ignore[assignment]
    hi_raw: np.ndarray = stats["hi_raw"]  # type: ignore[assignment]
    hi_ema: np.ndarray = stats["hi_ema"]  # type: ignore[assignment]
    slopes: np.ndarray = stats["slopes"]  # type: ignore[assignment]
    hi_thr = float(stats["hi_threshold"])
    slope_thr = float(stats["slope_threshold"])
    hi_base_mean = float(stats["hi_baseline_mean"])
    n_base = int(stats["n_base"])
    n_cal = int(stats["n_cal"])
    cal_end = n_base + n_cal
    m_consec = int(cfg["ims_consecutive_alarms"])

    fixed_trig = consecutive_runs(hi_ema > hi_thr, m_consec)
    slope_trig = consecutive_runs(np.isfinite(slopes) & (slopes > slope_thr) & (hi_ema > hi_base_mean), m_consec)
    first_fixed = int(cal_end + np.flatnonzero(fixed_trig[cal_end:])[0])
    first_slope = int(cal_end + np.flatnonzero(slope_trig[cal_end:])[0])
    if first_fixed != 527 or first_slope != 538:
        raise RuntimeError(f"Expected IMS Stage 4 indices (527, 538), got ({first_fixed}, {first_slope})")

    exclude = set(int(i) for i in cfg["ims_exclude_snapshots"])
    paths = [p for i, p in enumerate(snapshot_paths(ROOT / str(cfg["ims_snapshot_directory"]))) if i not in exclude]
    ch = int(cfg["ims_channel"])
    n_snaps = len(paths)
    n_samples = int(cfg["ims_sampling_rate"])
    if n_snaps != len(X):
        raise RuntimeError(f"IMS snapshot count mismatch: {n_snaps} vs {len(X)}")

    wav_bin = ROOT / "data/processed/ims_ch1_waveforms.bin"
    with wav_bin.open("wb") as handle:
        handle.write(struct.pack("<II", n_snaps, n_samples))
        for p in paths:
            sig = load_channel(p, ch).astype(np.float32)
            if sig.size != n_samples:
                raise RuntimeError(f"{p.name}: expected {n_samples}, got {sig.size}")
            handle.write(sig.tobytes())

    exp_bin = ROOT / "data/processed/ims_ch1_expected.bin"
    with exp_bin.open("wb") as handle:
        handle.write(struct.pack("<Ifff", n_snaps, slope_thr, hi_thr, hi_base_mean))
        for t in range(n_snaps):
            s_valid = 1 if np.isfinite(slopes[t]) else 0
            s_val = float(slopes[t]) if s_valid else 0.0
            handle.write(X[t].astype(np.float32).tobytes())
            handle.write(
                struct.pack(
                    "<fffiii",
                    float(hi_raw[t]),
                    float(hi_ema[t]),
                    s_val,
                    s_valid,
                    int(fixed_trig[t]),
                    int(slope_trig[t]),
                )
            )
    print(
        f"Wrote {n_snaps} IMS Ch 1 waveforms ({wav_bin}) & expectations ({exp_bin}; "
        f"first_fixed={first_fixed}, first_slope={first_slope})"
    )


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    export_cwru_all774(cfg_path)
    export_ims_ch1(cfg_path)


if __name__ == "__main__":
    main()

