"""Create window-level CWRU DE-signal features.

This offline script keeps its output lightweight: windowing and the seven
time-domain / four spectral features are simple enough to reproduce in C for an
ESP32. SciPy is used here only for offline anti-aliasing and descriptive moments.

Resampling strategy (single-step, source → target):
  48 kHz normal → 12 kHz  : decimate(q=4)
  48 kHz normal → 3200 Hz : resample_poly(up=2, down=30)
  12 kHz fault  → 3200 Hz : resample_poly(up=4, down=15)
  12 kHz fault  → 12 kHz  : no-op
"""

from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, resample_poly
from scipy.stats import kurtosis, skew


RECORDINGS = (
    ("normal_0.mat",   "normal"),
    ("IR007_0.mat",    "inner_race_007"),
    ("B007_0.mat",     "ball_007"),
    ("OR007@6_0.mat",  "outer_race_007"),
    ("IR014_0.mat",    "inner_race_014"),
    ("B014_0.mat",     "ball_014"),
    ("OR014@6_0.mat",  "outer_race_014"),
    ("IR021_0.mat",    "inner_race_021"),
    ("B021_0.mat",     "ball_021"),
    ("OR021@6_0.mat",  "outer_race_021"),
)
EPSILON = np.finfo(float).eps


def _parse_scalar(value: str) -> Any:
    """Parse the intentionally small YAML subset used by config.yaml."""
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"\'')


def load_config(path: Path) -> dict[str, Any]:
    """Load this project's scalar, inline-list, and one-level mapping YAML."""
    config: dict[str, Any] = {}
    active_mapping: dict[str, Any] | None = None
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(raw_lines):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  "):
            if active_mapping is None or ":" not in line:
                raise ValueError(f"Unsupported config line: {raw_line}")
            key, value = line.strip().split(":", 1)
            active_mapping[key] = _parse_scalar(value)
            continue
        key, value = line.split(":", 1)
        parsed = _parse_scalar(value)
        if parsed is None:
            following = ""
            for candidate in raw_lines[line_number + 1 :]:
                following = candidate.split("#", 1)[0].rstrip()
                if following.strip():
                    break
            if following.startswith("  "):
                active_mapping = {}
                config[key] = active_mapping
            else:
                active_mapping = None
                config[key] = None
        else:
            active_mapping = None
            config[key] = parsed
    return config


def de_signal_and_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read exactly one drive-end array and expose non-private MAT metadata."""
    mat = loadmat(path)
    public = {key: value for key, value in mat.items() if not key.startswith("__")}
    de_keys = [key for key in public if key.lower().endswith("_de_time")]
    if len(de_keys) != 1:
        raise ValueError(f"{path.name}: expected one *_DE_time variable, found {de_keys}")
    signal = np.asarray(public[de_keys[0]], dtype=float).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"{path.name}: DE signal must be a non-empty vector; got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"{path.name}: DE signal contains NaN or infinity")
    return signal, public


def resample_to(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Single-step resample from source_rate to target_rate with anti-aliasing.

    Exact ratios used (per Stage-2 spec):
      48000 → 12000 : decimate(q=4)         [integer factor]
      48000 →  3200 : resample_poly(up=2, down=30)
      12000 → 12000 : no-op
      12000 →  3200 : resample_poly(up=4, down=15)
    """
    if source_rate == target_rate:
        return signal
    if source_rate == 48000 and target_rate == 12000:
        return decimate(signal, 4, ftype="fir", zero_phase=True)
    if source_rate == 48000 and target_rate == 3200:
        return resample_poly(signal, 2, 30)
    if source_rate == 12000 and target_rate == 3200:
        return resample_poly(signal, 4, 15)
    # Fallback for any other ratio via Fraction reduction
    ratio = Fraction(target_rate, source_rate)
    if ratio.denominator == 1:
        return decimate(signal, ratio.numerator, ftype="fir", zero_phase=True)
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def matched_rate_signal(
    signal: np.ndarray, filename: str, label: str, source_rate: int, base_rate: int, target_rate: int | None
) -> tuple[np.ndarray, int]:
    """Single-step resample from source_rate directly to the pipeline output rate."""
    out_rate = target_rate if target_rate is not None else base_rate
    signal = resample_to(np.asarray(signal, dtype=float), source_rate, out_rate)
    return signal, out_rate


def window_signal(signal: np.ndarray, size: int, overlap: float):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("window_size must be positive and overlap must be in [0, 1).")
    hop = int(round(size * (1 - overlap)))
    if hop <= 0:
        raise ValueError("overlap produces a zero-length hop")
    for start in range(0, signal.size - size + 1, hop):
        yield signal[start : start + size]


def extract_features(window: np.ndarray, rate: int, bands: list[list[float]]) -> dict[str, float]:
    # Mean-centre the window before all feature computation (Stage-2 spec).
    window = window - window.mean()
    rms = float(np.sqrt(np.mean(window * window)))
    deviation = float(np.std(window))
    if rms <= EPSILON:
        crest = 0.0
    else:
        crest = float(np.max(np.abs(window)) / rms)
    if deviation <= EPSILON:
        kurt = 0.0
        skewness = 0.0
    else:
        kurt = float(kurtosis(window, fisher=False, bias=False))
        skewness = float(skew(window, bias=False))
    frequencies = np.fft.rfftfreq(window.size, d=1.0 / rate)
    power = np.abs(np.fft.rfft(window)) ** 2 / (window.size**2)
    features = {
        "rms": rms,
        "std": deviation,
        "peak_to_peak": float(np.ptp(window)),
        "crest_factor": crest,
        "kurtosis": kurt,
        "skewness": skewness,
    }
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies <= high if high == bands[-1][1] else frequencies < high)
        features[f"band_energy_{int(low)}_{int(high)}hz"] = float(np.sum(power[mask]))
    return {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in features.items()}


def validate_bands(bands: list[list[float]], rate: int) -> None:
    if not bands or any(len(band) != 2 or band[0] < 0 or band[0] >= band[1] for band in bands):
        raise ValueError("fft_bands_hz must contain ascending [low, high] pairs")
    if max(band[1] for band in bands) > rate / 2:
        raise ValueError(f"fft_bands_hz exceeds the {rate / 2:g} Hz Nyquist frequency")


def print_class_summary(rows: list[dict[str, Any]], feature_names: list[str]) -> None:
    print("Per-class feature summary (mean):")
    for label in sorted({row["label"] for row in rows}):
        class_rows = [row for row in rows if row["label"] == label]
        values = ", ".join(
            f"{name}={np.mean([float(row[name]) for row in class_rows]):.6g}" for name in feature_names
        )
        print(f"  {label} (windows={len(class_rows)}): {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--rate-mode",
        choices=("configured", "original"),
        default="configured",
        help="Use config target_rate, or retain the matched sampling_rate for the original-rate CSV.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    base_rate = int(config["sampling_rate"])
    target_rate = None if args.rate_mode == "original" or config.get("target_rate") is None else int(config["target_rate"])
    output_rate = target_rate or base_rate
    bands = config["fft_bands_hz"]
    validate_bands(bands, output_rate)

    available = [(name, label) for name, label in RECORDINGS if (args.input_dir / name).is_file()]
    if not available:
        names = ", ".join(name for name, _ in RECORDINGS)
        raise SystemExit(f"Error: no requested MAT files found in {args.input_dir}: {names}")

    source_rates = config.get("source_sampling_rates", {})
    rows: list[dict[str, Any]] = []
    for filename, label in available:
        signal, _ = de_signal_and_metadata(args.input_dir / filename)
        source_rate = int(source_rates.get(filename, base_rate))
        signal, rate = matched_rate_signal(signal, filename, label, source_rate, base_rate, target_rate)
        for window_idx, window in enumerate(window_signal(signal, int(config["window_size"]), float(config["overlap"]))):
            rows.append({"file": filename, "label": label, "window_idx": window_idx, **extract_features(window, rate, bands)})

    if not rows:
        raise ValueError("Present recordings are shorter than one configured window")
    feature_names = list(extract_features(np.zeros(int(config["window_size"])), rate, bands))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"features_{output_rate}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "label", "window_idx", *feature_names])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} windows to {output_path}")
    print_class_summary(rows, feature_names)


if __name__ == "__main__":
    main()
