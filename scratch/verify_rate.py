"""
Verify sampling rate of normal_0.mat via shaft-frequency spectral check,
using IR007_0 and B007_0 (known 12 kHz) as controls.
Also compute true min, max, mean for all 10 files.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import welch

RAW = Path("data/raw/cwru")

FILES_ALL = [
    "normal_0.mat", "IR007_0.mat", "B007_0.mat", "OR007@6_0.mat",
    "IR014_0.mat",  "B014_0.mat",  "OR014@6_0.mat",
    "IR021_0.mat",  "B021_0.mat",  "OR021@6_0.mat",
]

# ── 1. True stats (real min / max / mean) ─────────────────────────────────────
print("=" * 80)
print("TRUE SIGNAL STATISTICS (computed, not assumed)")
print("=" * 80)
print(f"{'File':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'RMS':>10} {'Kurt':>8}")
print("-" * 80)

for fname in FILES_ALL:
    p = RAW / fname
    if not p.exists():
        print(f"{fname:<20} MISSING")
        continue
    mat = loadmat(p)
    de_key = next(k for k in mat if not k.startswith("__") and k.lower().endswith("_de_time"))
    s = np.asarray(mat[de_key], dtype=float).squeeze()
    mean  = float(np.mean(s))
    std   = float(np.std(s, ddof=0))
    smin  = float(np.min(s))          # true minimum, independent of max
    smax  = float(np.max(s))          # true maximum, independent of min
    rms   = float(np.sqrt(np.mean(s**2)))
    kurt  = float(np.mean((s - mean)**4) / (std**4 + 1e-30))
    print(f"{fname:<20} {mean:>10.4f} {std:>10.4f} {smin:>10.4f} {smax:>10.4f} {rms:>10.4f} {kurt:>8.3f}")

# ── 2. Shaft-frequency spectral check ─────────────────────────────────────────
print()
print("=" * 80)
print("SHAFT-FREQUENCY SPECTRAL CHECK")
print("Expected shaft freq = RPM/60 ≈ 29.9 Hz (1796 RPM)")
print("Test: which assumed rate (12000 or 48000 Hz) places the dominant")
print("      low-freq spectral peak closest to shaft freq?")
print("=" * 80)

CANDIDATES = [
    ("normal_0.mat",  "UNKNOWN — under test"),
    ("IR007_0.mat",   "known 12 kHz (control)"),
    ("B007_0.mat",    "known 12 kHz (control)"),
    ("OR007@6_0.mat", "known 12 kHz (control)"),
]

for fname, label in CANDIDATES:
    p = RAW / fname
    mat = loadmat(p)
    de_key = next(k for k in mat if not k.startswith("__") and k.lower().endswith("_de_time"))
    s = np.asarray(mat[de_key], dtype=float).squeeze()

    rpm_key = next((k for k in mat if "rpm" in k.lower()), None)
    rpm = float(np.asarray(mat[rpm_key]).flat[0]) if rpm_key else 1796.0
    shaft_hz = rpm / 60.0

    N = s.size
    # Use Welch PSD for robust peak finding, fs=1 (bin units), then rescale
    # We work with raw FFT for exact bin-to-Hz mapping under two hypotheses
    fft_mag = np.abs(np.fft.rfft(s)) / N
    freqs_bins = np.fft.rfftfreq(N)   # 0..0.5 in units of fs

    # Search window: 10–80 Hz in physical terms
    # Under each hypothesis, convert Hz → normalised bin fraction
    results = {}
    for assumed_fs in (12000, 48000):
        lo = 10.0 / assumed_fs
        hi = 80.0 / assumed_fs
        mask = (freqs_bins >= lo) & (freqs_bins <= hi)
        if not mask.any():
            results[assumed_fs] = (None, None, None)
            continue
        peak_idx = np.argmax(fft_mag[mask])
        peak_bin_frac = freqs_bins[mask][peak_idx]
        peak_hz = peak_bin_frac * assumed_fs
        error_hz = abs(peak_hz - shaft_hz)
        results[assumed_fs] = (peak_hz, error_hz, fft_mag[mask][peak_idx])

    print(f"\n  {fname}  ({label})")
    print(f"    RPM={rpm:.0f}  shaft_freq={shaft_hz:.3f} Hz")
    for fs, (peak_hz, err, amp) in results.items():
        if peak_hz is None:
            print(f"    @ {fs} Hz assumed: no peak found in window")
        else:
            verdict = "✓ MATCHES" if err < 2.0 else ("close" if err < 5.0 else "✗ MISMATCH")
            print(f"    @ {fs:>5} Hz assumed: dominant low-freq peak = {peak_hz:6.2f} Hz  "
                  f"(error {err:.2f} Hz)  {verdict}")

    # Also find the single strongest peak in 0-100 Hz under each assumption
    # for a secondary cross-check
    print(f"    -- Top-3 peaks in 5–150 Hz (12 kHz assumption) --")
    mask2 = (freqs_bins >= 5/12000) & (freqs_bins <= 150/12000)
    if mask2.any():
        sub = fft_mag[mask2]
        sub_freqs = freqs_bins[mask2] * 12000
        top3 = np.argsort(sub)[::-1][:3]
        for i, idx in enumerate(top3):
            print(f"      [{i+1}] {sub_freqs[idx]:.2f} Hz  amp={sub[idx]:.6f}")

