"""Inspect raw 98/99/100.mat files, perform spectral shaft-freq check, rename to normal_1-3.mat."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

RAW_DIR = Path("data/raw/cwru")

# CWRU Normal Baseline motor speeds (RPM) at load 0,1,2,3 HP → motor ~1797,1772,1750,1730 rpm
# File 97=0HP, 98=1HP, 99=2HP, 100=3HP
FILE_MAP = {
    "98.mat": ("normal_1.mat", 1772),  # 1 HP → ~1772 RPM
    "99.mat": ("normal_2.mat", 1750),  # 2 HP → ~1750 RPM
    "100.mat": ("normal_3.mat", 1730), # 3 HP → ~1730 RPM
}

CANDIDATE_RATES = [48000, 12000]


def shaft_check(signal: np.ndarray, source_rate: int, nominal_rpm: int) -> tuple[float, float]:
    """Find the dominant spectral peak near the shaft frequency.
    Returns (peak_freq, expected_shaft_freq)."""
    expected_shaft_hz = nominal_rpm / 60.0
    n = len(signal)
    # Look in ±15 Hz window around expected shaft freq
    freqs = np.fft.rfftfreq(n, d=1.0 / source_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    low = max(expected_shaft_hz - 15, 1.0)
    high = expected_shaft_hz + 15
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return float("nan"), expected_shaft_hz
    peak_idx = np.argmax(spectrum[mask])
    peak_freq = freqs[mask][peak_idx]
    return peak_freq, expected_shaft_hz


def inspect_file(src_name: str, dest_name: str, nominal_rpm: int) -> None:
    src = RAW_DIR / src_name
    if not src.exists():
        print(f"  MISSING: {src}")
        return

    mat = loadmat(src)
    public = {k: v for k, v in mat.items() if not k.startswith("__")}
    de_keys = [k for k in public if k.lower().endswith("_de_time")]
    if len(de_keys) != 1:
        print(f"  ERROR: expected 1 *_DE_time key, found {de_keys}")
        return
    key = de_keys[0]
    signal = np.asarray(public[key], dtype=float).squeeze()
    n = signal.size

    print(f"\n{'='*60}")
    print(f"File     : {src_name}  →  {dest_name}")
    print(f"MAT var  : {key}")
    print(f"Samples  : {n:,}")
    print(f"File size: {src.stat().st_size / 1e6:.2f} MB")
    print(f"True min : {np.min(signal):.6f}")
    print(f"True max : {np.max(signal):.6f}")
    print(f"Mean     : {np.mean(signal):.6f}")
    print(f"Std      : {np.std(signal):.6f}")
    print(f"RMS      : {np.sqrt(np.mean(signal**2)):.6f}")
    from scipy.stats import kurtosis as kurt_fn
    print(f"Kurtosis : {kurt_fn(signal, fisher=False):.4f}  (Gaussian=3)")

    print(f"\nShaft-freq check (nominal {nominal_rpm} RPM = {nominal_rpm/60:.2f} Hz):")
    for rate in CANDIDATE_RATES:
        expected_n_for_rate = round(n * rate / 48000)  # expected samples if at this rate
        # Try the signal at face value at this rate
        peak, expected = shaft_check(signal, rate, nominal_rpm)
        err = abs(peak - expected)
        print(f"  Assuming {rate} Hz: peak at {peak:.2f} Hz, expected {expected:.2f} Hz, error {err:.2f} Hz")

    # Duration makes sense at 48 kHz → ~3.6 s (97/98/100) or ~7.2 s (99 is double-length)
    for rate in CANDIDATE_RATES:
        dur = n / rate
        print(f"  Duration if {rate} Hz: {dur:.3f} s")

    # Decision
    peak_48, exp_48 = shaft_check(signal, 48000, nominal_rpm)
    peak_12, exp_12 = shaft_check(signal, 12000, nominal_rpm)
    err_48 = abs(peak_48 - exp_48)
    err_12 = abs(peak_12 - exp_12)
    if err_48 < err_12:
        confirmed_rate = 48000
        confidence = "CONFIRMED 48 kHz (error {:.2f} Hz vs {:.2f} Hz at 12k)".format(err_48, err_12)
    else:
        confirmed_rate = 12000
        confidence = "CONFIRMED 12 kHz (error {:.2f} Hz vs {:.2f} Hz at 48k)".format(err_12, err_48)
    print(f"  → {confidence}")

    # Rename
    dest = RAW_DIR / dest_name
    if dest.exists():
        print(f"\nAlready renamed: {dest_name} exists, skipping rename.")
    else:
        src.rename(dest)
        print(f"\nRenamed {src_name} → {dest_name}")

    print(f"Source rate: {confirmed_rate}")
    print(f"Config entry: {dest_name}: {confirmed_rate}")


def main() -> None:
    print("Inspecting CWRU Normal Baseline files 98, 99, 100")
    for src_name, (dest_name, rpm) in FILE_MAP.items():
        inspect_file(src_name, dest_name, rpm)
    print("\nDone. Update config.yaml source_sampling_rates with the confirmed rates above.")


if __name__ == "__main__":
    main()

